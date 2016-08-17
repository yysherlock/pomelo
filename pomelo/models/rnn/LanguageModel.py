#from pomelo.models.BasicModel import BasicModel
from .BasicModel import BasicModel

class LanguageModel(BasicModel):
  """Abstracts a Tensorflow graph for learning language models.

  Adds ability to do embedding.
  """

  def add_embedding(self):
    """Add embedding layer. that maps from vocabulary to vectors.
    """
    raise NotImplementedError("Each Model must re-implement this method.")
