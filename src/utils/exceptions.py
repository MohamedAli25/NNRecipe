class ShapeError(Exception):
    def __init__(self, given_shape, required_shape):
        super(ShapeError, self).__init__(
            "Required Shape is {} but the given shape is {}".format(required_shape, given_shape)
        )
