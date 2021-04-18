import array

class CardValue:
    def __init__(self, value:int):
        assert value >= 0
        assert value <= 12

        self.value = value

    def __eq__(self, other):
        if isinstance(other, CardValue):
            return self.value == other.value
        return NotImplemented

class PlayedCards:
    def __init__(self, value : CardValue, count : int):
        assert count >= 1

        self.value = value
        self.count = count

class Hand:
    def __init__(self):
        self._cards = array.array('i', [0,0,0,0,0,0,0,0,0,0,0,0,0] )

    def add(self,x,y):
        print (self._cards)
        print (len(self._cards))
        return x + y