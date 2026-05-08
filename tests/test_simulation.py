import numpy as np

from functional_fitted_q.envs import Pendulum_data_generator


def test_toy_simulation_shapes_and_seed():
    np.random.seed(123)
    first = Pendulum_data_generator(episode_num=3, cycle_num=2, x_length=10)
    np.random.seed(123)
    second = Pendulum_data_generator(episode_num=3, cycle_num=2, x_length=10)

    assert len(first) == 6
    assert first["Subject"].nunique() == 3
    assert first["FunctionalData"].map(len).eq(10).all()
    assert first["Reward"].between(0, 1).all()
    assert np.allclose(first.iloc[0]["FunctionalData"], second.iloc[0]["FunctionalData"])
