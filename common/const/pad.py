""" Index for padding """
PAD_ID = -1  #: PADDING VALUE

""" Infinity values (we use 1E10 for numerical stability) """
NEG_INF = float('-inf')  #: EXACT NEGATIVE INFINITY
NEG_INF_SAFE = -1E10  #: PSEUDO-INFINITY (-) FOR NUMERICAL STABILITY
POS_INF = float('inf')  #: EXACT POSITIVE INFINITY
POS_INF_SAFE = 1E10  #: PSEUDO-INFINITY (+) FOR NUMERICAL STABILITY
FLOAT_NAN = float('NaN')  #: NOT-A-NUMBER VALUE

UNEXPLAINED_NUMBER = 'skip'
