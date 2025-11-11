import pysindy as ps

def create_sindy_model(x, dx, u, dt, threshold, degree):
    dim_x = x[0].shape[1]
    dim_u = u[0].shape[1]

    feature_lib = ps.PolynomialLibrary(degree=degree, include_bias=False)
    '''
    parameter_lib = ps.PolynomialLibrary(degree=1, include_interaction=False, include_bias=False)

    lib = ps.ParameterizedLibrary(
        feature_library=feature_lib,
        parameter_library=parameter_lib,
        num_features=dim_x,
        num_parameters=dim_u,
    )
    '''
    opt = ps.STLSQ(threshold=threshold, max_iter=10)
    model = ps.SINDy(
        feature_library=feature_lib,
        optimizer=opt,
        feature_names=[f"x{i}" for i in range(dim_x)],
        discrete_time=False,
    )
    print("Training SINDy model...", flush=True)

    model.fit(x, t=dt, x_dot=dx, multiple_trajectories=True)

    return model
