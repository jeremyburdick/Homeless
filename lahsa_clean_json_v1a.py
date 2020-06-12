import json

def cleanup(ifile, ofile):

    f = open(ifile)


    j = json.load(f)

    def expand(c,name):

    #
    # expect
    #
    #  census code, estimated total, census code, community name, unsheltered, sheltered, ??
    #
    # example:
    #
    # ['701100a', 27, '701100a', 'Brentwood', 3, 27, 24]

        if c[-1] in ('Unincorporated','City'): c.pop()

#        print(c)
#        c = [x if x is str and not x.isnumeric() else float(x) for x in c]

        for i, x in enumerate(c,0):
            if type(x) == str and x.find('.') > 0:
                c[i] = float(x)

#        print(c)

        z = len(c)

        if z < 3 or (type(c[1]) == str and str.find(c[1],'.') == -1):
            # change
            # census, census, ...   TO
            # census, 0, census, ...

            c.insert(1, 0)

            print("11111", c)

            expand(c, name)

        elif z < 4 or type(c[3]) != str:
            # change
            # census, X, census, ...   TO
            # census, X, census, "", ...

            c.insert(3, name)

            print("22222", c)

            expand(c, name)

        elif z < 5:

            # unsheltered with estimated total
            # fill sheltered with 0

            c.append(c[1])
            c.append(0)

            print("33333", c)

        elif z < 6:
            # fill unsheltered with 0
            c.append(0)

            print("44444", c)


    n = []

    for i in range(0,7):
        n.append(0)

    g = []

    def printcensus(pz):

        lastname = ""

        for d in j:
            for y in j[d]:
                for x in y:
                    for c in y[x]:
                        if (c is None):
                           continue

                        z = len(c)

                        if (z > 6):
                            print("************ WTF!!!!!!!!")

                        print("-----", c)

                        expand(c, lastname)

    #                    print("xxxxx", c)

                        n[z-1]+=1

                        c.insert(0, x[1:])

                        g.append(c)

                        lastname = c[4]
                        print(lastname)




    printcensus(i)

    o = open(ofile, "w")

    print("Year,CensusCode,TotalEst,CensusCode2,Name,Unsheltered,Sheltered,Total", file = o )

    for c in g:
        for x in c:
            print("%s," % x, end='', file=o)
        print(file=o)

    for i in range(0,6):
        print(i, n[i])


for y in [2016,2017,2018,2019]:
    fi = '_dump_lacoc_%i.json.jqproc' % y
    print(fi)
    fo = fi + '.clean'
    cleanup(fi, fo)


#for i in range(0,7):
#    print(i+1, n[i])
