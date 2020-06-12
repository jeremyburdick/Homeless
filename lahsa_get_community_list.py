import requests as req
import json
import gzip

## all stored queries have embedded community = Venice and year = 2018

## queries by community name

qryfile = "examples/qry_dropdown_0a.json"

def readfile(x):
    f = open(x)
    s = f.read()
    f.close()
    return s

qry = readfile(qryfile)

h = {'Accept': 'application/json, text/plain, */*',
     'ActivityId': '14b7e49a-a5a3-45e8-8d39-e25721da7c0a',
     'Content-Type': 'application/json;charset=UTF-8',
     'DNT': '1',
     'Origin': 'https://app.powerbi.com',
     'Referer': 'https://app.powerbi.com/view?r=eyJrIjoiZTkyNDg3ZTMtNmYxMi00MWZkLTgyMjctM2Q1M2I2ZTgzM2UwIiwidCI6IjBiYWU1NDliLTUyZDgtNGEzYi1hYTE5LWQ1MDY2MmIzMDg5NyIsImMiOjZ9',
     'RequestId': '06a49592-366e-4d4e-4493-97c429931ab5',
     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36',
     'X-PowerBI-ResourceKey': 'e92487e3-6f12-41fd-8227-3d53b6e833e0',
     'X-PowerBI-User-GroupId': ''
     }

def getpost(qry, textout):
    r = req.post('https://wabi-west-us-api.analysis.windows.net/public/reports/querydata', data=qry, headers=h, stream=True)

    raw = r.raw.read(25000)

    f = open(textout + ".gz", 'wb')
    f.write(raw)
    f.close()

    ret = gzip.decompress(raw).decode('utf-8')

    f = open(textout, 'w')
    f.write(ret)
    f.close()

    return ret


def get_neighborhood_list():

    rt = "Acton"

    ic = False

    xx = []

    npass = 1

    while not ic and npass < 4:

        nlst = "v2_lahsa_result_community_lst_" + str(npass) + ".json"

        print(nlst)

        qlst = qry.replace('Industry', rt).replace('2018', '2019')

        rlst = getpost(qlst, nlst)

        j = json.loads(rlst)

        lst = j["results"][0]["result"]["data"]["dsr"]["DS"][0]["PH"][0]["DM0"]

        print(lst)

        x = list(map(lambda z: z["G0"], lst))
        print(x)

        xx.extend(x)

        ic  = j["results"][0]["result"]["data"]["dsr"]["DS"][0]["IC"]

        if not ic:
            rt  = j["results"][0]["result"]["data"]["dsr"]["DS"][0]["RT"][0][0]
        else:
            rt = ""

        print(ic)
        print(rt)

        rt = rt.replace("'","")
        print(rt)


        npass += 1

#    print(xx)
    return xx


zz = get_neighborhood_list()
print(zz)

f = open('lahsa_community_list.txt','w')
f.writelines(map(lambda x: x + '\n', zz))
f.close()


#    for yi in y:
#        print()

        # nbrk = "v2_lahsa_result_community_brk" + yi + "_" + community_no_space + ".json"

        # qbrk = community_qry_brk.replace('Venice', ni).replace('2018', yi)

        # print("%s %s" % (ni, yi))

        # getpost(qbrk, nbrk)




exit()









#print(r.headers)

