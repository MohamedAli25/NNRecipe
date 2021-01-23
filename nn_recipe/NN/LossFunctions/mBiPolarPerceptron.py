from .__loss_function import LossFunction
import numpy as np


class MClassBipolarPerceptron(LossFunction):
    def _compute_loss(self, Y, Y_hat):
        correct_value = Y_hat[0, np.argmax(Y, axis=1).reshape(-1, 1)]
        # print(Y)
        # print(np.argmax(Y, axis=1).reshape(-1, 1))
        # print(Y_hat)
        # print((Y_hat - correct_value).T)
        # print(np.maximum(np.zeros_like(Y_hat), Y_hat - correct_value))
        # print(np.max(np.maximum(np.zeros_like(Y_hat), Y_hat - correct_value), axis=1))
        # print(np.max(np.maximum(np.zeros_like(Y_hat), Y_hat - correct_value), axis=1).reshape((-1, 1)))
        # print(correct_value)
        return np.max(np.maximum(np.zeros_like(Y_hat), Y_hat - correct_value), axis=1).reshape((-1, 1))

    def _compute_local_grad(self, Y, Y_hat):
        correct_value_index = np.argmax(Y, axis=1).reshape(-1, 1)
        correct_value = Y_hat[0, correct_value_index]
        max_value = np.argmax(Y_hat - correct_value, axis=1).reshape(-1, 1)
        print(Y_hat.T)
        print((Y_hat - correct_value).T)
        print(correct_value_index)
        print((max_value == correct_value_index).squeeze())
        print(np.where(max_value != correct_value))
        out = np.zeros_like(Y_hat)
        # print("out", out)
        # out = out[(max_value != correct_value_index).squeeze(), :] + 1
        out2 = out[np.where(max_value != correct_value), :] + 1
        print("out", out2)
        # self.__softmax_value[np.argmax(Y, axis=0), range(self.__softmax_value.shape[1])] -= 1
        # return self.__softmax_value

