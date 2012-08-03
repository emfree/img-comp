import hierarchy2
src = hierarchy2.data.src


vars = []
mdists = []

for i in range(1, 512, 2):
    print i
    for j in range(1, 512, 2):
        nbrs = [src[i-1, j], src[i, j+1], src[i-1, j], src[i, j-1]]
        val = src[i, j]
        avg = sum(nbrs) / 4.
        mdist = val - avg
        var = sum((nbr - avg)**2 for nbr in nbrs)
        mdists.append(mdist)
        vars.append(var)

vars = np.array(vars)
mdists = np.array(mdists)
