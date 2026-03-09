import numpy as np

N = 5000000

a = np.random.normal(-1, 1, N)
b = np.random.normal(-1, 1, N)
c = np.random.normal(-1, 1, N)
d = np.random.normal(-1, 1, N)

trace = a + d
det = a*d - b*c

dis = trace**2 - 4*det

saddle = det < 0
spiral = dis < 0
node = (dis > 0) & (det > 0)

sbl_spiral = spiral & (trace < 0)
usbl_spiral = spiral & (trace > 0)

sbl_node = node & (trace < 0)
usbl_node = node & (trace > 0)

print("Saddle:", np.mean(saddle))
print("Stable Spiral:", np.mean(sbl_spiral))
print("Unstable Spiral:", np.mean(usbl_spiral))
print("Stable Node:", np.mean(sbl_node))
print("Unstable Node:", np.mean(usbl_node))
