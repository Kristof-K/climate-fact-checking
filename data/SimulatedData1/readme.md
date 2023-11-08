simple data to check whether the network can learn simple patterns:

- tokens at start and mid is always "a", and at the end is "b"

neural network has to learn whether masked word is at the at the end or not
--> difficult as RNNs work under assumption of time-invariance, i.e. if sth appears
    later cannot be perceived