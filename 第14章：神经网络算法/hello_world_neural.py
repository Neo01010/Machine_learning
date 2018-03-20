from sklearn.neural_network import MLPClassifier

X = [[0.,0.],[1.,1.]]
Y = [0, 1]

clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, Y)

print(clf.n_layers_)
print(clf.n_iter_)
print(clf.loss_)
print(clf.out_activation_)
print(clf.predict([[2,2],[-1,-2]]))