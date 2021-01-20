from nn_recipe.NN.LossFunctions.crossEntropy import CrossEntropyLoss

c = CrossEntropyLoss()

print(c(1, 0))
print(c.local_grad)