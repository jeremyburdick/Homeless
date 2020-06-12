from urllib.request import urlopen
from urllib.parse import urlparse

import pandas as pd
import json
import os
import pickle
import numpy as np
from area import area

import plotly
import plotly.express as px

#print(px.colors.named_colorscales())

##print(px.colors.hex_to_rgb('#ffffff'))

#print(px.colors.colorscale_to_colors(px.colors.cmocean.thermal))

#print(len(px.colors.cmocean.thermal))
#x = px.colors.make_colorscale(px.colors.cmocean.thermal, [0,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.95,0.97,0.99,1])
#print(x)

##print(px.colors.colorscale_to_scale(colorscale_to_scale))




def urlbasename(url): return os.path.basename(urlparse(url).path)

def fromCacheOrReq(url):
    file = urlbasename(url)
    if not os.path.exists(file):
        with urlopen(url) as req:
            open(file, 'wb').write(req.read())

    return open(file,'r')

            

def fromPickOrParse(url, loadfunc, dumpfunc, parsefunc, **kwargs):
    file = urlbasename(url) + '.pickle'
    if os.path.exists(file):
        return loadfunc(open(file,'rb'))
    else: 
        i = fromCacheOrReq(url)
        o = parsefunc(i, **kwargs)
        dumpfunc(o, open(file, 'wb'))
        return o

    #TRACTA


tracts: dict = fromPickOrParse('https://raw.githubusercontent.com/arcee123/GIS_GEOJSON_CENSUS_TRACTS/master/06.geojson', 
        pickle.load,
        pickle.dump,
        json.load)

df: pd.DataFrame = fromPickOrParse('nhgis0002_ds172_2010_tract.csv', 
                     pd.read_pickle, 
                     pd.to_pickle,
                     pd.read_csv,
                     dtype=dict(TRACTA=str))
#                     ,usecols=['TRACTA','H7V001'])

print(len(tracts['features']))
tracts_LAC = tracts
tracts_LAC['features'] = [f for f in tracts_LAC['features'] if f['properties']['COUNTYFP'] == '037']
#for f in tracts_LAC['features']:
#    f.update({'id':f['properties']['TRACTCE']})

print(len(tracts_LAC['features']))

#areas = {f['id']:area(f['geometry']) for f in tracts_LAC['features']}

SQ_METERS_TO_SQ_MILES = 3.86102e-7
areas = {f['properties']['TRACTCE']:area(f['geometry']) * SQ_METERS_TO_SQ_MILES for f in tracts_LAC['features']}

dfAreas = pd.DataFrame.from_dict(areas, orient = 'index', columns = ['area'])


df = df[df.COUNTYA == 37]
df = df[['TRACTA','H7V001']]
df = df.set_index('TRACTA')
df.columns = ['population']
print(len(df.index))
print(df)


df = df.merge(dfAreas, left_index=True, right_index=True, how='inner')
print(df)

df['density'] = df.population / df.area


homeless = 'C:\\workflow\\homeless\\lahsa_counts\\_dump_ALL.csv'

h = pd.read_csv(homeless,dtype={'CensusCode':str})
h.drop
h['TRACTA'] = [x[0:6] for x in h.CensusCode]

h = h[['TRACTA','Name','Year','Unsheltered','Sheltered']]

hname = pd.DataFrame(h.groupby(['TRACTA'])['Name'].apply(lambda x: '/'.join(set(x))))
print(hname)

hh = h.groupby(['TRACTA','Year']).sum()
print(hh)

hh = hh.merge(hname, left_index=True, right_index=True)



print(hh)
print(df)


df = df.rename_axis('TRACTA')
hh = hh.reset_index(level=['Year'])
print(hh)
df = hh.join(df, how='inner')
#df = df.merge(hh, left_on=['TRACTA'], right_on=['TRACTA'], how='inner')
df['hdense'] = df.Unsheltered / df.area
df['hrate'] = df.Unsheltered / df.population

chartVar = 'hdense'
chartLabel = {'Unsheltered Homeless Density (per sq mile)',chartVar}
chartData = df[chartVar]


hsub = df[df.Year == 2016]


def eqdist(x, y, numbins):
    tot = sum(x)
    runx = 0
    curbin = 0
    binsize = tot / numbins
    curcut = binsize
    bins = [0]
    for i, v in enumerate(x, start=0):
        runx += v
        if runx >= curcut:
            curbin += 1
            curcut = binsize * (curbin + 1)
            bins.append(y[i])

    if len(bins) < numbins:    
        bins.append(y[-1])

    return bins

#colors = px.colors.cmocean.thermal
#colors = px.colors.sequential.Viridis
#colors = px.colors.sequential.Inferno
colors = px.colors.sequential.YlOrRd

max_data = max(hsub[chartVar])

#df = df.sort_values(by=chartVar)
#bins = eqdist(df.Unsheltered, df[chartVar], len(colors)-1)


bb = np.linspace(0,1,len(colors))
hsub2 = hsub[hsub.hdense > 0]
bins = hsub2[chartVar].quantile(bb)

print(bins)

#base = max_data ** (1/float(len(colors)-1))
#bins = [0]
#for i in range(1,len(colors)):
#    bins.append(base**i)

scale = list(np.divide(bins, max_data))
scale[0] = 0
scale[len(colors)-1] = 1
print(scale)
#scale = np.linspace(0,1,len(colors))
#print(scale)

#colorscale = px.colors.make_colorscale(colors, scale)
colorscale = [list(tup) for tup in zip(scale, colors)]
print(colorscale)

#exit(0)

pp = df.groupby(pd.cut(df[chartVar],bins)).agg(['mean','count','sum'])
print(pp)


#nonzero = df[df[chartVar] > 0]
#qs = [0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999,1]
#qq = nonzero[chartVar].quantile(qs)

#pp = df.groupby(pd.cut(df.hrate,qq)).agg(['mean','count','sum'])
#print(pp)

z = df.sort_values(by='hdense', ascending=False)
print(z.head(30)[['Name','population','area','hdense','Unsheltered','Sheltered']])

z = z.sort_values(by='hrate', ascending=False)
z = z[z.population > 500]
print(z.head(30)[['Name','population','area','hrate','Unsheltered','Sheltered']])

#exit(0)

rMin = 0.5
rMax = 0.99

#df.reset_index(level=['Year'])

df = df.sort_values(by='Year')

q = df[chartVar].quantile([rMin,rMax])
print(q)

def doChart(df, year = None): 

    kwargs = {}
    if year:
        df = df[df.Year == year]
    else:
        kwargs = dict(animation_frame='Year')


    

    fig =  px.choropleth_mapbox(df, 
        geojson=tracts, 
        featureidkey='properties.TRACTCE',
        locations=df.index, 
        color=chartVar,
    #    color_continuous_scale="cividis",
        color_continuous_scale=colorscale,
    #    range_color=(q[rMin], q[rMax]),
        range_color=(0, 1000),
        mapbox_style="carto-positron",
        zoom=10, center = {"lat": 34.0522, "lon": -118.2437},
        opacity=0.2,
        labels=chartLabel,
        **kwargs
        )

    cb = fig.data[0].colorbar


    cb.tickmode = 'array'
    cb.ticktext = ['0','10','100','1,000','10,000']
    cb.tickvals = [0,10,100,1000,10000]

#    fig.update_layout(coloraxis_colorbar=cb)

    if 'animation_frame' in kwargs:
        duration = 1000
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},transition=dict(duration=duration))
        fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = duration

    fig.show()

#def makeGIF():
#    import ImageSequence
#    import Image
#    import gifmaker
#    sequence = []

##    im = Image.open(....)

#    # im is your original image
#    frames = [frame.copy() for frame in ImageSequence.Iterator(im)]

#    # write GIF animation
#    fp = open("out.gif", "wb")
#    gifmaker.makedelta(fp, frames)
#    fp.close()


doChart(df, 2019)








exit(0)


