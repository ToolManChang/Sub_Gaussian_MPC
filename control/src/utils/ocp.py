import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_sim import AcadosSim
from .model import (
    export_integrator_model,
    export_n_integrator_model,
    export_integrator_ode_model_with_discrete_rk4,
    export_integrator_model_discrete_rk4,
    export_integrator_model_discrete_euler,
    export_linear_model,
)


def mpc_const_expr(model, x_dim, n_order, params, model_x):
    xg = ca.SX.sym("xg", x_dim)
    w = ca.SX.sym("w", 1, 1)

    p_lin = ca.vertcat(xg, w)
    a_list = params["env"]['constraints']["a"]
    b_list = params["env"]['constraints']["b"]
    centers = params["env"]["constraints"]["mean"]
    covs = params["env"]["constraints"]["cov"]
    shift = params['env']['constraints']['shift']
    radius = params['env']['constraints']['radius']
    
    # compute constraints
    circle_const_list = []
    if params["env"]['constraints']['circle']:
        for i in range(len(centers)):
            center = np.array(centers[i])
            cov = np.array(covs[i])
            circle_const_list.append(
                1. 
                - (model_x - center).reshape((1, -1)) 
                @ ca.inv(cov + params["tube"]["tightening"]['circle']**2) # 
                @ (model_x - center).reshape((-1, 1))
            )


    linear_const_list = []
    a = np.array(a_list) # (N, 2)
    b = np.array(b_list) # (N, 1)
    if params["env"]['constraints']['polygon']:
        for i in range(len(a)):
            try:
                linear_const_list.append(a[i].reshape((1, -1)) @ model_x.reshape((-1, 1)) - b[i] + params["tube"]["tightening"]['polygon'][i]) #
            except:
                linear_const_list.append(a[i].reshape((1, -1)) @ model_x.reshape((-1, 1)) - b[i] + params["tube"]["tightening"]['polygon'])

    exp_const_list = []
    if params["env"]['constraints']['exponential']:
        exp_const_list.append(
            (model_x[1] + params["tube"]["tightening"]['exponential'] - radius) - ca.exp(-(model_x[0] - shift))
        )
        # exp_const_list.append(
        #     (-model_x[1] + params["tube"]["tightening"] - radius) - ca.exp(-(model_x[0] - shift))
        # )
    funnel_const_list = []
    if params["env"]['constraints']['funnel']['if_funnel']:
        shift = params["env"]['constraints']['funnel']["shift"]
        dz = params["env"]['constraints']['funnel']["dz"]
        dy = params["env"]['constraints']['funnel']["dy"]
        dx = params["env"]['constraints']['funnel']["dx"]
        l = params["env"]['constraints']['funnel']["screw_len"]
        cons = params["env"]['constraints']['funnel']["constraint"]
        tighten = params["tube"]["tightening"]['funnel']
        funnel_const_list.append(
            (model_x[2] / dz) ** 2
            + (model_x[1] / dy) ** 2
            + 2 * tighten / dz * np.sqrt((model_x[2] / dz) ** 2 + (model_x[1] / dy) ** 2)
            + (tighten / dz) **2
            - ca.exp((-model_x[0] / (dx)) - shift) - cons
        )
        funnel_const_list.append(
            ((model_x[2] + l * ca.sin(model_x[3]) * ca.sin(model_x[4])) / dz) ** 2
            + ((model_x[1] + l * ca.sin(model_x[3]) * ca.cos(model_x[4])) / dy) ** 2
            + 2 * (tighten / dz) * np.sqrt(((model_x[2] + l * ca.sin(model_x[3]) * ca.sin(model_x[4])) / dz) ** 2 
                                    + ((model_x[1] + l * ca.sin(model_x[3]) * ca.cos(model_x[4])) / dy) ** 2)
            + (tighten / dz)**2
            - ca.exp((-(model_x[0] + l * ca.cos(model_x[3])) / (dx)) - shift) - cons
        ) 
    
    const_list = tuple(circle_const_list + linear_const_list + exp_const_list + funnel_const_list)

    model.con_h_expr = ca.vertcat(
        *const_list
    )
    model.con_h_expr_e = ca.vertcat(
        *const_list
    )
    
    model.p = p_lin
    return model, w, xg


def mpc_cost_expr(ocp, model_x, model_u, x_dim, w, xg, params):
    P = np.array(params['env']['P'])
    # qx = np.diag(np.ones(x_dim))
    goal = np.array(params["env"]["goal_loc"])
    xg = np.array(goal)  # p[0]
    w = params["optimizer"]["w"]
    qx = np.diag(np.array(params['env']['Q']))
    r = np.diag(np.array(params['env']['R']))
    # qx = np.eye(x_dim)
    # r = np.eye(r.shape[0])
    
    # cost
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.model.cost_expr_ext_cost = (model_x[:x_dim] - xg).T @ qx @ (
        model_x[:x_dim] - xg
    ) + model_u.T @ r @ model_u
    ocp.model.cost_expr_ext_cost_e = (
        model_x[:x_dim] - xg
        ).T @ P @ (
            model_x[:x_dim] - xg
        ) 
    
        # ocp.model.cost_expr_ext_cost = (model_x[:x_dim] - xg).T @ qx @ (
        #     model_x[:x_dim] - xg
        # ) + model_u.T @ (q) @ model_u
        # ocp.model.cost_expr_ext_cost_e = (
        #     (model_x[:x_dim] - xg).T @ qx @ (model_x[:x_dim] - xg)
    

    # TODO: Uncomment for slack variables
    # ocp.constraints.idxsh = np.array([1])
    # ocp.cost.zl = 1e2 * np.array([1])
    # ocp.cost.zu = 1e1 * np.array([1])
    # ocp.cost.Zl = 1e1 * np.array([1])
    # ocp.cost.Zu = 1e1 * np.array([1])

    return ocp


def mpc_const_val(ocp, params, x_dim, n_order):

    ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
    ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
    ocp.constraints.idxbu = np.arange(params["optimizer"]['u_dim'])

    lbx = np.array(params["optimizer"]["x_min"])
    ubx = np.array(params["optimizer"]["x_max"])

    x0 = np.zeros(ocp.model.x.shape[0])
    x0[:x_dim] = np.array(params["env"]["start_loc"])  # np.ones(x_dim)*0.72
    ocp.constraints.x0 = x0.copy()

    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])

    # TODO: For complicated constraints
    num_constraints = 0
    if params["env"]["constraints"]["circle"]:
        num_constraints += len(params["env"]["constraints"]["mean"])
    if params["env"]["constraints"]["polygon"]:
        num_constraints += len(params["env"]["constraints"]["a"])
    if params["env"]["constraints"]["exponential"]:
        num_constraints += 1
    if params["env"]["constraints"]["funnel"]['if_funnel']:
        num_constraints += 2
    
    constraint_val = 0.0
    ocp.constraints.lh = -1e6 * np.ones((num_constraints,))
    ocp.constraints.uh = np.ones((num_constraints,)) * constraint_val
    ocp.constraints.lh_e = -1e6 * np.ones((num_constraints,))
    ocp.constraints.uh_e = np.ones((num_constraints,)) * constraint_val

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp


def mpc_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["Tf"]
    ocp.solver_options.sim_method_num_steps = 60

    ocp.solver_options.qp_solver_warm_start = 1
    # set options
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = 1.0e-1
    ocp.solver_options.integrator_type = "DISCRETE"  #'IRK'  # IRK
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    # ocp.solver_options.tol = 1e-6
    ocp.solver_options.regularize_method = 'CONVEXIFY'
    ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
    # ocp.solver_options.alpha_min = 1e-2
    # ocp.solver_options.__initialize_t_slacks = 0
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.levenberg_marquardt = 1e-1
    # ocp.solver_options.print_level = 2
    ocp.solver_options.qp_solver_iter_max = 400
    # ocp.solver_options.regularize_method = 'MIRROR'
    # ocp.solver_options.exact_hess_constr = 0
    # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
    # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
    # ocp.solver_options.eps_sufficient_descent = 5e-1
    # params = {'globalization': ['MERIT_BACKTRACKING', 'FIXED_STEP'],
    #       'line_search_use_sufficient_descent': [0, 1],
    #       'globalization_use_SOC': [0, 1]}
    return ocp


def export_mpc_ocp(params):
    ocp = AcadosOcp()
    name_prefix = (
        params["algo"]["type"]
        + "_env_"
        + str(params["env"]["name"])
        + "_i_"
        + str(params["env"]["i"])
        + "_"
    )
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    u_dim = params["optimizer"]["u_dim"]
    # model = export_integrator_model_discrete_rk4(
    #     name_prefix + "mpc", params, n_order, x_dim
    # )
    # model = export_integrator_model_discrete_euler(
    #     name_prefix + "mpc", params, n_order, x_dim
    # )
    model = export_linear_model(
        name_prefix + "mpc", params, n_order, u_dim, x_dim
    )
    # model = export_n_integrator_model(name_prefix + "mpc", n_order, x_dim)
    # model = export_integrator_ode_model_with_discrete_rk4(
    #     name_prefix + "mpc", params, n_order, x_dim
    # )
    # model = export_n_integrator_model(name_prefix + "mpc", n_order, x_dim)
    # model = export_integrator_ode_model_with_discrete_rk4(
    #     name_prefix + "mpc", params, n_order, x_dim
    # )
    model_u = model.u[:u_dim]
    model_x = model.x[:x_dim]
    # model_z = model.u[-x_dim:] # if extra variables required

    # TODO: If we want to add constraints, uncomment and use it, currently only state, input constraints work
    model, w, xg = mpc_const_expr(model, x_dim, n_order, params, model_x)

    ocp.model = model

    ocp = mpc_cost_expr(ocp, model_x, model_u, x_dim, w, xg, params)

    ocp = mpc_const_val(ocp, params, x_dim, n_order)

    ocp = mpc_set_options(ocp, params)

    return ocp
