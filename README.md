# DOG - Diagonalization Over Gaussians

Produces nearly complete, orthogonal, diagonal bases by diagonalizing the X operator over a set of Gaussians defined by a width and a (possibly unstructured) distribution of function centers.

dog.py is 1D

dog2.py is 2D

hexagonal_zigzag_basis.py is over hexagonally-arranged function centers

random_centers_basis.py is over randomly-arranged function centers

hydrogen_projected_gaussian_basis.py projects onto the orthogonal subspace to the Hydrogen orbitals up to the principal quantum number n then applies the same construction.
Note that there is weird behavior near the origin. This shows the need for PAW, which I'll be including next.
