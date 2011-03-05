from qitensor.sage import *

HA = qudit('A', 3)
HB = qudit('B', 3)
HAbar = qudit('Abar', 3)
HBbar = qudit('Bbar', 3)
Hc = qudit('c', 5)
Ha = qudit('a', 5)
Hb = qudit('b', 5)

U = ((HA*HB).H*(HAbar*HBbar)).eye()
U[{HA.H: 1, HB.H: 2, HAbar: 1, HBbar: 2}] = 0
U[{HA.H: 2, HB.H: 1, HAbar: 1, HBbar: 2}] = 1
U[{HA.H: 1, HB.H: 2, HAbar: 2, HBbar: 1}] = 1
U[{HA.H: 2, HB.H: 1, HAbar: 2, HBbar: 1}] = 0

print '*** U'
print U

Utilde = U.transpose(HA*HBbar)

print '*** Utilde'
print Utilde

E0 = (HAbar * HA.H * Hc.H).array()
E0[{Hc.H: 0}] = [[1,0,0], [0,0,0], [0,0,0]]
E0[{Hc.H: 1}] = [[0,0,0], [0,1,0], [0,0,0]]
E0[{Hc.H: 2}] = [[0,0,0], [0,0,0], [0,1,0]]
E0[{Hc.H: 3}] = [[0,0,0], [0,0,1], [0,0,0]]
E0[{Hc.H: 4}] = [[0,0,0], [0,0,0], [0,0,1]]

print '*** E0'
print E0

F0 = (E0.transpose(HA).pinv() * Utilde).transpose(Hc*HBbar)

print '*** F0'
print F0

psi = (Ha*Hb.H).eye().transpose(Hb)

print '*** psi'
print psi

U2 = (E0 * (Hc*Ha.H).eye()) * (F0 * (Hc*Hb.H).eye()) * psi

print '*** U2'
print U2
