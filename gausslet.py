from G10 import *
from plot1d import *

arr1 = G10(jval, 0.5)
plot_jnp_1d(jval, arr1)
arr2 = G10(jval - 20)
# plot_jnp_1d(jval, arr1)

print(arr1 @ arr2)
print(arr1 @ arr1)

print(arr1 @ (jval * arr1))
print(arr2 @ (jval * arr2))
print(arr1 @ (jval * arr2))