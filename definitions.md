# y variables
y1 = X13 - X14
y2 = X12 - X10

# w variables
w1  = X2 + X4 - X8
w2  = X1 - X5 - X6
w3  = X6 + X7
w4  = X14 + X15
w5  = y2 + X16
w6  = X10 + X11
w7  = X9 + y1
w8  = X9 - X8
w9  = X7 - X11
w10 = X6 - X7
w11 = X2 - X3

# Multiplications (m variables)
m1 = (-w1 + X3) @ (X8 + X11).T
m2 = (w2 + X7) @ (X15 + X5).T
m3 = (-X2 + X12) @ w5.T
m4 = (X9 - X6) @ w7.T
m5 = (X2 + X11) @ (X15 - w3).T
m6 = (X6 + X11) @ (w3 - X11).T
m7 = X11 @ w3.T
m8 = X2 @ (w3 - w4 + w5).T
m9 = X6 @ (w7 - w6 + w3).T
m10 = (w1 - X3 + X7 + X11) @ X11.T
m11 = (X5 + w10) @ X5.T
m12 = (w11 + X4) @ X8.T
m13 = (-w2 + X3 - w9) @ X15.T
m14 = (-w2) @ (w7 + w4).T
m15 = w1 @ (w6 + w5).T
m16 = (X1 - X8) @ (X9 - X16).T
m17 = X12 @ (-y2).T
m18 = X9 @ y1.T
m19 = (-w11) @ (-X15 + X7 + X8).T
m20 = (X5 + w8) @ X9.T
m21 = X8 @ (X12 + w8).T
m22 = (-w10) @ (X5 + w9).T
m23 = X1 @ (X13 - X5 + X16).T
m24 = (-X1 + X4 + X12) @ X16.T
m25 = (X9 + X2 + X10) @ X14.T
m26 = (X6 + X10 + X12) @ X10.T

# Recursive calls (s variables)
s1 = X1 @ X1.T
s2 = X2 @ X2.T
s3 = X3 @ X3.T
s4 = X4 @ X4.T
s5 = X13 @ X13.T
s6 = X14 @ X14.T
s7 = X15 @ X15.T
s8 = X16 @ X16.T

# Intermediate sums (z variables)
z1 = m7 - m11 - m12
z2 = m1 + m12 + m21
z3 = m3 + m17 - m24
z4 = m2 + m11 + m23
z5 = m5 + m7 + m8
z6 = m4 - m18 - m20
z7 = m6 - m7 - m9
z8 = m17 + m18

# Final matrix entries (C variables)
C11 = s1 + s2 + s3 + s4
C12 = m2 - m5 - z1 + m13 + m19
C13 = z2 + z3 + m15 + m16
C14 = z4 - z3 - z5 + m13
C22 = m1 + m6 - z1 + m10 + m22
C23 = z2 - z6 + z7 + m10
C24 = z4 + z6 + m14 + m16
C33 = m4 - z7 - z8 + m26
C34 = m3 + z5 + z8 + m25
C44 = s5 + s6 + s7 + s8