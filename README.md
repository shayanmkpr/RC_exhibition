#Model Architecture and Node characteristics

The reservoir includes DHO nodes that are connected to each other and to themselves. These connections
have been initialized randomly. The input is given to each node via a weighted connection, and the outputs
are assigned to 10 nodes representing the 10 different classes of the dataset.
In the following equations, the dynamics of the reservoir are introduced:
I rec (t) = Whhy(t) + bhh + v · x(t)
Iext (t) = Wihs(t) + bih
And the following equations explain the dynamic of each node in this reservoir.
xt+1 = xt + hyt+1
(
 )
1
yt+1 = yt + h[α · tanh
 √ Irec (t) + Iext (t) − 2γ · yt − ω 2 · xt]
n
Each node has its own values of α, γ, and ω. This variability in parameters introduces heterogeneity to
the model. To achieve this, each node’s parameters were randomly distributed around a central value derived
from experiences with the homogeneous network. A deviation of 0.1 from the central value was assigned to
each node’s parameters.
The following graphs illustrate how these parameters have been assigned for 16 nodes:

#Training

For the training of this model, two major approaches has been implemented.
• First the input and output layers of the model where trained using Back propagation through time
(BPTT), while the reservoir had fixed weights that were initially assigned randomly.
• Second, while updating the input and output weights as explained in the previous training approach, the
hidden weights of the reservoir, which were constant in the previous training method, will be updated
according the the Hebbian Rule.

The hebbian learning has been implemented according to the following equation,
∆Wij = σλh aij r(xi(t), xj (t))
Where
∑n
i=1(xi − x)(yi − y)
r = √∑n
 √∑n
i=1 (xi − x)
2
 i=1 (yi − y)
2