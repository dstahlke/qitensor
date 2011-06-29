#!/usr/bin/python

import unittest
from qitensor.experimental.cartan_decompose_impl import *

class CartanTests(unittest.TestCase):
    def random_unitary(self, dim):
        (Q, _R) = linalg.qr(np.random.rand(dim, dim) + 1j*np.random.rand(dim, dim))
        return np.matrix(Q)

    def testMagicBasis(self):
        #print "Making sure magic basis really is a basis."
        mb = MAGIC_BASIS
        assert np.allclose(mb.H * mb, np.eye(4))
        assert np.allclose(mb * mb.H, np.eye(4))

    def testXformToMagic(self):
        #print "Testing locally_transform_to_magic_basis."
        mb = MAGIC_BASIS
        for i in range(20):
            U = self.random_unitary(4)
            UT = mb.H * (mb * U * mb.H).T * mb
            (ew, psi) = linalg.eig(UT * U)
            assert np.allclose(np.log(ew).real, 0)

            (UA, UB, gamma) = locally_transform_to_magic_basis(psi)
            UAB = bipartite_op_tensor_product(UA, UB)

            assert np.allclose(UAB * psi * np.matrix(np.diag(np.exp(1j*gamma))), mb.H)

    def testRandomUnitary(self):
        #print "Testing unitary_to_cartan with random matrices."
        for i in range(20):
            U = self.random_unitary(4)

            (UA, UB, VA, VB, alpha) = unitary_to_cartan(U)

            Ud = unitary_from_cartan(alpha)
            UAB = bipartite_op_tensor_product(UA, UB)
            VAB = bipartite_op_tensor_product(VA, VB)
            assert np.allclose(UAB * Ud * VAB, U)

    def testRandomAlpha(self):
        #print "Testing going from alpha to unitary back to alpha."
        for i in range(20):
            alpha_in = np.random.rand(3) * np.pi/2 - np.pi/4
            U = unitary_from_cartan(alpha_in)
            (UA, UB, VA, VB, alpha_out) = unitary_to_cartan(U)
            # because the results are not unique, just compare the sorted absolute
            # values
            assert np.allclose(np.sort(abs(alpha_in)), np.sort(abs(alpha_out)))

    def testSpecificUnitary(self):
        #print "Testing unitary_to_cartan with some specific matrices."
        phase = np.pi/3
        for U in [
            np.matrix(np.diag([1, 1, 1, np.exp(1j*phase)])),
            np.matrix(np.diag([np.exp(1j*x) for x in [1e-6, 2e-6, 3e-6, phase]])),
            np.matrix([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]),
        ]:
            (UA, UB, VA, VB, alpha) = unitary_to_cartan(U)

            Ud = unitary_from_cartan(alpha)
            UAB = bipartite_op_tensor_product(UA, UB)
            VAB = bipartite_op_tensor_product(VA, VB)
            assert np.allclose(UAB * Ud * VAB, U)

def suite():
    return unittest.TestSuite(map(unittest.TestLoader().loadTestsFromTestCase, [
        CartanTests,
    ]))

if __name__ == "__main__":
    unittest.TextTestRunner().run(suite())
