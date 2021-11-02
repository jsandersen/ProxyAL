from enum import Enum, auto

class Datasets(Enum):
    imdb = auto()
    app_store = auto()
    
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Datasets[s]
        except KeyError:
            raise ValueError()
            
            
class Encodings(Enum):
    sbert = auto()
    elmo = auto()
    sif = auto()
    iftif = auto()
    bert = auto()
    
    def __str__(self):
        return self.name

    @staticmethod
    def from_string(s):
        try:
            return Encodings[s]
        except KeyError:
            raise ValueError()