# DOG - Diagonalization Over Gaussians

Produces nearly complete, orthogonal, diagonal bases by diagonalizing the position operator over a set of Gaussians defined by a width and a (possibly unstructured) distribution of function centers.
Because function centers now are up to us to choose, we can choose them according to our application. 
A good metric for tuning widths and function centers might be the error in representing polynomials in the basis. Ritz-Galerkin or the normal equation may be used for this purpose.

! I'm currently reading into PAW and ECP for smoothing out the interacting electron subspace.

The code is made with Codex, and it shouldn't be used for production applications but mere demonstration.

dog.py is 1D

dog2.py is 2D

hexagonal_basis.py is over hexagonally-arranged function centers

random_centers_basis.py is over randomly-arranged function centers

hydrogen_projected_gaussian_basis.py projects onto the orthogonal subspace to the Hydrogen orbitals up to the principal quantum number n then applies the same construction.
Note that there is weird behavior near the origin. This shows the need for PAW, which I'll be including next.
