import warnings

from floodlight import Pitch


class BaseModel:
    """Base class for all models.

    Attributes
    ----------
    pitch: Pitch, optional
        Some models are require pitch information, so the corresponding Pitch object is
        handled during initialization.
    """

    def __init__(self, pitch: Pitch = None):

        if pitch is not None:
            self.check_pitch(pitch)
            self._pitch = pitch

    def __str__(self):
        return f"Floodlight {self.__class__.__name__}"

    @property
    def is_fitted(self) -> bool:
        """Returns ``True`` if all model parameters (those with a trailing underscore)
        are fitted (i.e. not None), and ``False`` otherwise."""
        fitted = all(
            [
                vars(self)[v] is not None
                for v in vars(self)
                if (v.endswith("_") and not v.startswith("__"))
            ]
        )

        return fitted

    @staticmethod
    def check_pitch(pitch: Pitch):
        """
        Performs a series of checks on a Pitch object and raises warnings if the pitch
        configuration may affect computation results.

        Parameters
        ----------
        pitch: Pitch
            Pitch object the checks are performed on.
        """
        # check if metrical system
        if not pitch.is_metrical:
            warnings.warn(
                "Model initialized with non-metrical pitch. Results may be distorted, "
                "use at your own risk.",
                category=RuntimeWarning,
            )
