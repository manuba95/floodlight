import pytest
import numpy as np

from floodlight.models.geometry import (
    CentroidModel,
    NearestMateModel,
    NearestOpponentModel,
)


# Test fit function of CentroidModel with different xIDs excluded
@pytest.mark.unit
def test_centroid_model_fit(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = CentroidModel()
    model.fit(xy)
    centroid1 = model._centroid_
    model.fit(xy, exclude_xIDs=[0])
    centroid2 = model._centroid_
    model.fit(xy, exclude_xIDs=[0, 1])
    centroid3 = model._centroid_

    # Assert
    assert np.array_equal(centroid1, np.array(((1.5, -1), (1.25, 0.5))))
    assert np.array_equal(centroid2, np.array(((2, -2), (1, 0.5))))
    assert np.array_equal(
        centroid3,
        np.array(((np.nan, -2), (1, 1))),
        equal_nan=True,
    )


# Test centroid function of CentroidModel
@pytest.mark.unit
def test_centroid(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = CentroidModel()
    model.fit(xy)
    centroid = model.centroid()

    # Assert
    assert np.array_equal(centroid, np.array(((1.5, -1), (1.25, 0.5))))


# Test centroid_distance function of CentroidModel
@pytest.mark.unit
def test_centroid_distance(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = CentroidModel()
    model.fit(xy)
    distance = model.centroid_distance(xy)

    # Assert
    assert np.array_equal(
        np.round(distance, 3),
        np.array(((2.062, 1.118, np.nan), (np.nan, np.nan, 0.559))),
        equal_nan=True,
    )


# Test stretch_index function of CentroidModel
@pytest.mark.unit
def test_stretch_index(example_xy_object_geometry) -> None:
    # Arrange
    xy = example_xy_object_geometry
    xy.framerate = 20

    # Act
    model = CentroidModel()
    model.fit(xy)
    stretch_index1 = model.stretch_index(xy)
    stretch_index2 = model.stretch_index(xy, axis="x")
    stretch_index3 = model.stretch_index(xy, axis="y")

    # Assert
    assert np.array_equal(np.round(stretch_index1, 3), np.array((1.59, 0.559)))
    assert np.array_equal(np.round(stretch_index2, 3), np.array((0.5, 0.25)))
    assert np.array_equal(np.round(stretch_index3, 3), np.array((1.333, 0.5)))
    assert stretch_index1.framerate == 20


# Test fit function of NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_fit(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = NearestMateModel()
    model.fit(xy)

    # Assert
    assert model._pairwise_distances_ is not None
    assert isinstance(model._pairwise_distances_, np.ndarray)
    assert model._pairwise_distances_.shape == (2, 3, 3)


# Test distance_to_nearest_mate function of NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_distance_to_nearest_mate(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()

    # Assert
    expected = np.array(((np.sqrt(10), np.sqrt(10), np.nan), (np.nan, np.nan, np.nan)))
    assert np.array_equal(dtnm.property, expected, equal_nan=True)


# Test team_spread function of NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_team_spread(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry

    # Act
    model = NearestMateModel()
    model.fit(xy)
    spread = model.team_spread()

    # Assert
    expected = np.array(((np.sqrt(10),), (np.nan,)))
    assert np.array_equal(spread.property, expected, equal_nan=True)


# Test framerate propagation for NearestMateModel
@pytest.mark.unit
def test_nearest_mate_model_with_framerate(example_xy_object_geometry):
    # Arrange
    xy = example_xy_object_geometry
    xy.framerate = 25

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()
    spread = model.team_spread()

    # Assert
    assert dtnm.framerate == 25
    assert spread.framerate == 25


# Test NearestMateModel with horizontal NaN slice
@pytest.mark.unit
def test_nearest_mate_model_horizontal_nan_slice(
    example_xy_object_geometry_horizontal_nan,
):
    # Arrange
    xy = example_xy_object_geometry_horizontal_nan

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()
    spread = model.team_spread()

    # Assert
    expected_dtnm = np.array(
        (
            (np.sqrt(8), np.sqrt(10), np.sqrt(8)),
            (np.nan, np.nan, np.nan),
            (np.sqrt(6.25), np.sqrt(6.25), np.sqrt(8)),
        )
    )
    assert np.array_equal(dtnm.property, expected_dtnm, equal_nan=True)
    assert np.isnan(spread.property[1, 0])


# Test NearestMateModel with single player
@pytest.mark.unit
def test_nearest_mate_model_single_player(example_xy_object_single_player):
    # Arrange
    xy = example_xy_object_single_player

    # Act
    model = NearestMateModel()
    model.fit(xy)
    dtnm = model.distance_to_nearest_mate()
    spread = model.team_spread()

    # Assert
    expected_dtnm = np.array(((np.nan,), (np.nan,), (np.nan,)))
    expected_spread = np.array(((np.nan,), (np.nan,), (np.nan,)))
    assert np.array_equal(dtnm.property, expected_dtnm, equal_nan=True)
    assert np.array_equal(spread.property, expected_spread, equal_nan=True)


# Test fit function of NearestOpponentModel
@pytest.mark.unit
def test_nearest_opponent_model_fit(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)

    # Assert
    assert model._pairwise_distances_ is not None
    assert isinstance(model._pairwise_distances_, np.ndarray)
    assert model._pairwise_distances_.shape == (2, 3, 3)


# Test distance_to_nearest_opponent function of NearestOpponentModel
@pytest.mark.unit
def test_nearest_opponent_model_distance_to_nearest_opponent(
    example_xy_objects_space_control,
):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno1 = np.array(
        ((30.0, 0.0, 10.0), (np.sqrt(1417), np.sqrt(146), np.sqrt(500)))
    )
    assert np.array_equal(dtno1.property, expected_dtno1)


# Test framerate propagation for NearestOpponentModel
@pytest.mark.unit
def test_nearest_opponent_model_with_framerate(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    assert dtno1.framerate == 20
    assert dtno2.framerate == 20


# Test NearestOpponentModel with NaN propagation
@pytest.mark.unit
def test_nearest_opponent_model_with_nan_propagation(example_xy_objects_space_control):
    # Arrange
    xy1, xy2 = example_xy_objects_space_control

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno2 = np.array(((30.0, 0.0, np.sqrt(116)), (31.0, np.nan, np.sqrt(146))))
    assert np.array_equal(dtno2.property, expected_dtno2, equal_nan=True)


# Test NearestOpponentModel with horizontal NaN slice
@pytest.mark.unit
def test_nearest_opponent_model_horizontal_nan_slice(
    example_xy_objects_horizontal_nan,
):
    # Arrange
    xy1, xy2 = example_xy_objects_horizontal_nan

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno1 = np.array(
        (
            (100.0, np.sqrt(8200), np.sqrt(6800)),
            (np.nan, np.nan, np.nan),
            (100.0, np.sqrt(8200), np.sqrt(6800)),
        )
    )
    expected_dtno2 = np.array(
        (
            (np.sqrt(6800), np.sqrt(8200), 100.0),
            (np.nan, np.nan, np.nan),
            (np.sqrt(6800), np.sqrt(8200), 100.0),
        )
    )
    assert np.array_equal(dtno1.property, expected_dtno1, equal_nan=True)
    assert np.array_equal(dtno2.property, expected_dtno2, equal_nan=True)


# Test NearestOpponentModel with single players
@pytest.mark.unit
def test_nearest_opponent_model_single_players(example_xy_objects_single_players):
    # Arrange
    xy1, xy2 = example_xy_objects_single_players

    # Act
    model = NearestOpponentModel()
    model.fit(xy1, xy2)
    dtno1, dtno2 = model.distance_to_nearest_opponent()

    # Assert
    expected_dtno1 = np.array(((100.0,), (100.0,), (100.0,)))
    expected_dtno2 = np.array(((100.0,), (100.0,), (100.0,)))
    assert np.array_equal(dtno1.property, expected_dtno1)
    assert np.array_equal(dtno2.property, expected_dtno2)
