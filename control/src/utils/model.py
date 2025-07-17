import casadi as ca
import numpy as np
from acados_template import AcadosModel


def export_integrator_model(name):
    model = AcadosModel()
    x = ca.SX.sym("x")
    x_dot = u = ca.SX.sym("x_dot")
    u = ca.SX.sym("u")

    model.f_expl_expr = u  # xdot=u
    model.f_impl_expr = x_dot - u  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name
    return model


def export_linear_model(name, params, n_order=4, u_dim=2, x_dim=2):
    model = AcadosModel()
    x = ca.SX.sym("x", n_order * x_dim)
    x_dot = ca.SX.sym("x_dot", n_order * x_dim)
    u = ca.SX.sym("u", u_dim)

    A = np.array(params['env']['A'])
    B = np.array(params['env']['B'])

    dT = params["optimizer"]["dt"]

    f_expl = A @ x + B @ u # * dT
    f_impl = x_dot - f_expl

    model.disc_dyn_expr = f_expl
    model.f_expl_expr = f_expl  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name
    return model


def export_n_integrator_model(name, n_order=4, x_dim=2):
    # x^n = A x + Bu
    model = AcadosModel()
    x = ca.SX.sym("x", n_order * x_dim)
    x_dot = ca.SX.sym("x_dot", n_order * x_dim)
    u = ca.SX.sym("u", x_dim)

    A = np.diag(np.ones((n_order - 1) * x_dim), x_dim)
    B = np.zeros((n_order * x_dim, x_dim))
    np.fill_diagonal(np.fliplr(np.flipud(B)), 1)

    f_expl = A @ x + B @ u
    f_impl = x_dot - f_expl

    model.f_expl_expr = f_expl  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name
    return model


def export_integrator_ode_model_with_discrete_rk4(name, params, n_order=4, x_dim=2):
    model = export_n_integrator_model(name, n_order, x_dim)
    # dT = ca.SX.sym('dt', 1)
    # T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    # model.x = ca.vertcat(x, T)
    # model.u = ca.vertcat(u, dT)
    # x = model.x
    # u = model.u
    # xdot = ca.vertcat(model.xdot, 1)
    # f_expl = ca.vertcat(model.f_expl_expr, 1)
    # model.f_expl_expr = f_expl
    # model.f_impl_expr = xdot - f_expl

    dT = params["optimizer"]["dt"]
    ode = ca.Function("ode", [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x, u)
    k2 = ode(x + dT / 2 * k1, u)
    k3 = ode(x + dT / 2 * k2, u)
    k4 = ode(x + dT * k3, u)
    xf = x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model


def export_integrator_model_discrete_rk4(name, params, n_order=4, x_dim=2):
    # x^n = A x + Bu
    model = AcadosModel()
    x = ca.SX.sym("x", n_order * x_dim)
    x_dot = ca.SX.sym("x_dot", n_order * x_dim)
    u = ca.SX.sym("u", x_dim)

    dT = params["optimizer"]["dt"]
    k1 = x + u
    k2 = x + dT / 2 * k1 + u
    k3 = x + dT / 2 * k2 + u
    k4 = x + dT * k3 + u
    xf = x + dT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # A = np.diag(np.ones((n_order - 1) * x_dim), x_dim)
    # B = np.zeros((n_order * x_dim, x_dim))
    # np.fill_diagonal(np.fliplr(np.flipud(B)), 1)

    # f_expl = A @ x + B @ u
    f_impl = x_dot - xf
    model.disc_dyn_expr = xf
    model.f_expl_expr = xf  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name

    return model


def export_integrator_model_discrete_euler(name, params, n_order=4, x_dim=2):
    # x^n = A x + Bu
    model = AcadosModel()
    x = ca.SX.sym("x", n_order * x_dim)
    x_dot = ca.SX.sym("x_dot", n_order * x_dim)
    u = ca.SX.sym("u", x_dim)

    dT = params["optimizer"]["dt"]

    xf = x + dT * u

    # A = np.diag(np.ones((n_order - 1) * x_dim), x_dim)
    # B = np.zeros((n_order * x_dim, x_dim))
    # np.fill_diagonal(np.fliplr(np.flipud(B)), 1)

    # f_expl = A @ x + B @ u
    f_impl = x_dot - xf
    model.disc_dyn_expr = xf
    model.f_expl_expr = xf  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name

    return model
