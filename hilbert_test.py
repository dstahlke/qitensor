#!/usr/bin/python

import unittest
import hilbert
from hilbert import qubit, qudit, indexed_space
from hilbert import MismatchedIndexSetError, DuplicatedSpaceError
from hilbert import BraKetMixtureError, HilbertIndexError, HilbertShapeError

class HilbertBasicTests(unittest.TestCase):
    def setUp(self):
        self.Ha = qubit('a')
        self.Hb = qubit('b')
        self.Hc = qubit('c')
        self.Hx = indexed_space('x', ['x', 'y', 'z'])

    def testStrings(self):
        self.assertEqual(str(self.Ha), '|a>')
        self.assertEqual(str(self.Ha.H), '<a|')
        self.assertEqual(str(self.Ha*self.Hb), '|a,b>')
        self.assertEqual(str(self.Ha*self.Hb.H), '|a><b|')

    def testAtomCompatibility(self):
        Ha2 = indexed_space('a', [0, 1])
        Ha3 = indexed_space('a', [0, 1, 2])
        # OK, same index set
        self.Ha * Ha2.H
        # Not OK, different index set
        self.assertRaises(MismatchedIndexSetError,
            lambda: self.Ha * Ha3.H)

        self.assertEqual(self.Ha, Ha2)
        self.assertRaises(MismatchedIndexSetError,
            lambda: self.Ha == Ha3)

    def testRedundant(self):
        Ha2 = indexed_space('a', [0, 1])
        self.assertRaises(DuplicatedSpaceError, lambda: self.Ha * self.Ha)
        self.assertRaises(DuplicatedSpaceError, lambda: self.Ha * Ha2)
        self.assertRaises(DuplicatedSpaceError, lambda: self.Ha.H * self.Ha.H)
        self.assertRaises(DuplicatedSpaceError, lambda: self.Ha.H * Ha2.H)
        A = (self.Ha * (self.Hb * self.Hx).H).array()
        B = (self.Ha * self.Hx * self.Hb).array()
        self.assertRaises(DuplicatedSpaceError, lambda: A*B)

    def testMulSpaces(self):
        A = (self.Ha * (self.Hb * self.Hx * self.Hc).H).array()
        B = (self.Hx * self.Hb).array()
        self.assertEqual(str((A*B).space), '|a><c|')
        A = (self.Ha * self.Hb.H).array()
        B = (self.Hx * self.Hb * self.Hc).array()
        self.assertEqual(str((A*B).space), '|a,c,x>')

    def testIndexing(self):
        Ha = self.Ha
        Hx = self.Hx

        Ma = Ha.array([4, 5])
        self.assertEqual(Ma[0], 4)
        self.assertEqual(Ma[1], 5)
        self.assertRaises(HilbertIndexError, lambda: Ma[2])

        Mx = Hx.array([4, 5, 6])
        self.assertEqual(Mx['x'], 4)
        self.assertEqual(Mx['y'], 5)
        self.assertEqual(Mx['z'], 6)
        self.assertRaises(HilbertIndexError, lambda: Mx[0])

        Max = (Ha * Hx).array([[1,2,3],[4,5,6]])

        self.assertEqual(Max[0, 'x'], 1)
        self.assertEqual(Max[1, 'x'], 4)
        self.assertEqual(Max[0, 'y'], 2)
        self.assertEqual(Max[0, :].nparray.shape, (3,))
        self.assertEqual(Max[0, :].space, Hx)
        self.assertEqual(Max[:, 'x'].space, Ha)

        self.assertEqual(Max[{Ha: 0, Hx: 'x'}], 1)
        self.assertEqual(Max[{Ha: 1, Hx: 'x'}], 4)
        self.assertEqual(Max[{Ha: 0, Hx: 'y'}], 2)
        self.assertEqual(Max[{Ha*Hx: (0, 'y')}], 2)
        self.assertEqual(Max[{Hx*Ha: (0, 'y')}], 2)
        self.assertEqual(Max[{Ha: 0}].nparray.shape, (3,))
        self.assertEqual(Max[{Ha: 0}].space, Hx)
        self.assertEqual(Max[{Hx: 'x'}].space, Ha)
        self.assertRaises(HilbertIndexError, lambda: Max[{Hx.H: 'x'}])

        self.assertNotEqual(Max, (Ha * Hx).array([[8,2,3],[9,5,6]]))
        Max[:, 'x'] = Ha.array([8, 9])
        self.assertEqual(Max, (Ha * Hx).array([[8,2,3],[9,5,6]]))
        Max[:, 'x'] = [11, 12]
        self.assertEqual(Max, (Ha * Hx).array([[11,2,3],[12,5,6]]))

        self.assertEqual(Max[{Hx: 'z'}][1], 6)
        Max[{Hx: 'z'}][1] = 13
        self.assertEqual(Max[{Hx: 'z'}][1], 13)

    def testShape(self):
        self.assertRaises(HilbertShapeError,
            lambda: self.Hx.array([4, 5]))

class HilbertOperTests(unittest.TestCase):
    def setUp(self):
        self.Ha = qubit('a')
        self.Hb = qubit('b')
        self.Hc = qubit('c')
        self.Hx = indexed_space('x', ['x', 'y', 'z'])
        self.epsilon = 1e-10

    def testIdentity(self):
        H = self.Ha * self.Hx
        self.assertEqual(H.eye(), (H*H.H).eye())
        # Spaces are not the same, but are the same size.  This is okay.
        (self.Ha.H * self.Hb).eye()
        self.assertRaises(HilbertShapeError, lambda: (self.Ha.H * self.Hx).eye())
        
    def testInverse(self):
        H = self.Ha * self.Hx
        H2 = H * H.H
        for i in range(20):
            M = H2.random_array() + 1j*H2.random_array()
            if abs(M.det()) > self.epsilon:
                MI = M.I
                self.failUnless( (M*MI - H2.eye()).norm() < self.epsilon )
                self.failUnless( (MI*M - H2.eye()).norm() < self.epsilon )
                self.failUnless( abs(M.det() * MI.det() - 1) < self.epsilon )
        
    def testTranspose(self):
        Ha = self.Ha
        Hb = self.Hb
        Hx = self.Hx

        M = (Ha * Ha.H).array()
        M[0, 1] = 5
        self.assertEqual(M.T[1, 0], 5)
        self.assertEqual(M.T, M.transpose())
        self.assertEqual(M.T, M.transpose(Ha))
        self.assertEqual(M.T, M.transpose(Ha.H))
        self.assertEqual(M.T, M.transpose(Ha * Ha.H))
        self.assertRaises(HilbertIndexError, lambda: M.transpose(Hb))

        M = (Ha * Hb * Hx.H).array()
        M[{Ha: 0, Hb: 1, Hx.H: 'z'}] = 5
        self.assertEqual(M.T[{Ha.H: 0, Hb.H: 1, Hx: 'z'}], 5)
        self.assertEqual(M.transpose(Ha)[{Ha.H: 0, Hb: 1, Hx.H: 'z'}], 5)
        self.assertEqual(M.transpose(Hx)[{Ha: 0, Hb: 1, Hx: 'z'}], 5)
        self.assertEqual(M.T, M.transpose())
        self.assertEqual(M.T, M.transpose(Ha*Hb*Hx))

class HilbertComplicatedTests(unittest.TestCase):
    def setUp(self):
        self.epsilon = 1e-10

    def testMapStateDuality(self):
        HA = qudit('A', 3)
        HB = qudit('B', 3)
        HAbar = qudit('Abar', 3)
        HBbar = qudit('Bbar', 3)
        Hc = qudit('c', 5)
        Ha = qudit('a', 5)
        Hb = qudit('b', 5)

        U = ((HA*HB).H*(HAbar*HBbar)).eye()
        self.assertEqual(str(U.space), '|Abar,Bbar><A,B|')
        U[{HA.H: 1, HB.H: 2, HAbar: 1, HBbar: 2}] = 0
        U[{HA.H: 2, HB.H: 1, HAbar: 1, HBbar: 2}] = 1
        U[{HA.H: 1, HB.H: 2, HAbar: 2, HBbar: 1}] = 1
        U[{HA.H: 2, HB.H: 1, HAbar: 2, HBbar: 1}] = 0

        Utilde = U.transpose(HA*HBbar)
        self.assertEqual(str(Utilde.space), '|A,Abar><B,Bbar|')

        E0 = (HAbar * HA.H * Hc.H).array()
        self.assertEqual(str(E0.space), '|Abar><A,c|')
        E0[{Hc.H: 0}] = [[1,0,0], [0,0,0], [0,0,0]]
        E0[{Hc.H: 1}] = [[0,0,0], [0,1,0], [0,0,0]]
        E0[{Hc.H: 2}] = [[0,0,0], [0,0,0], [0,1,0]]
        E0[{Hc.H: 3}] = [[0,0,0], [0,0,1], [0,0,0]]
        E0[{Hc.H: 4}] = [[0,0,0], [0,0,0], [0,0,1]]

        F0 = (E0.transpose(HA).pinv() * Utilde).transpose(Hc*HBbar)
        self.assertEqual(str(F0.space), '|Bbar><B,c|')
        self.failUnless( ((E0 * F0.transpose(Hc)) - U).norm() < self.epsilon )

        psi = (Ha*Hb.H).eye().transpose(Hb)
        self.assertEqual(str(psi.space), '|a,b>')

        U2 = (E0 * (Hc*Ha.H).eye()) * (F0 * (Hc*Hb.H).eye()) * psi
        self.assertEqual(str(U2.space), '|Abar,Bbar><A,B|')
        self.failUnless( (U2-U).norm() < self.epsilon )

if __name__ == "__main__":
    unittest.main()
