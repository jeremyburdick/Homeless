from urllib.request import urlopen
from urllib.parse import urlparse

import pandas as pd
import json
import os
import pickle
import numpy as np
import itertools

import nearest as near

from area import area
import shapely as sp
from shapely.geometry import Polygon
from shapely.ops import nearest_points
import geopandas as gp

import plotly
import plotly.graph_objects as go
import plotly.express as px
import math



GEOJSON_CALIFORNIA = 'https://raw.githubusercontent.com/arcee123/GIS_GEOJSON_CENSUS_TRACTS/master/06.geojson'
HOMELESS_COUNTS = 'C:\\workflow\\homeless\\lahsa_counts\\_dump_ALL.csv'
CENSUS_TRACT_POPULATIONS = 'nhgis0002_ds172_2010_tract.csv'



def urlbasename(url): return os.path.basename(urlparse(url).path)

def fromCacheOrReq(url, readmode='r'):
    file = urlbasename(url)
    if not os.path.exists(file):
        with urlopen(url) as req:
            open(file, 'wb').write(req.read())

    return open(file,readmode)

            

def fromPickOrParse(parsefunc, loadfunc = pickle.load, dumpfunc = pickle.dump, picklefile=None, url=None, urlcachereadmode='r', **kwargs):
    file = picklefile or urlbasename(url) + '.pickle'
    if os.path.exists(file):
        print("Loading %s... " % file, end='', flush=True)
        ret = loadfunc(open(file,'rb'))
        print("DONE.")
        return ret
    else: 
        i = None if url is None else fromCacheOrReq(url, urlcachereadmode) 
        o = parsefunc(i, **kwargs)
        dumpfunc(o, open(file, 'wb'))
        return o

def dfFromPickOrCSV(filename, picklefile=None, **kwargs):
    return fromPickOrParse(parsefunc = pd.read_csv,
                           loadfunc = pd.read_pickle,
                           dumpfunc = pd.to_pickle,
                           picklefile = picklefile,
                           url = filename,
                           kwargs = kwargs)

 

def eqdist(x, y, numbins):
    tot = sum(x)
    runx = 0
    curbin = 0
    binsize = tot / numbins
    curcut = binsize
    ybins = [0]
    xruns = [0]
    for i, v in enumerate(x, start=0):
        runx += v
        if runx >= curcut:
            curbin += 1
            curcut = binsize * (curbin + 1)
            ybins.append(y[i])
            xruns.append(runx)
            print(curbin, y[i], xruns[curbin], xruns[curbin]-xruns[curbin-1])

    if len(ybins) < numbins:    
        ybins.append(y[-1])

    return ybins

def loadCensusTractShapes(geojson):

    tracts: dict = fromPickOrParse(parsefunc=json.load, url=geojson)
    
    print("Found %i census tract shapes." % len(tracts['features']))

    tracts_LAC = tracts
    tracts_LAC['features'] = [f for f in tracts_LAC['features'] if f['properties']['COUNTYFP'] == '037']

    print("Found %i census tract shapes in LA County (COUNTYFP='037')." % len(tracts_LAC['features']))

    return tracts_LAC

def loadCensusTractPopulations(filename):

    df: pd.DataFrame = dfFromPickOrCSV(
                        filename = filename, 
                        dtype = dict(TRACTA=str))                     
    #                     ,usecols=['TRACTA','H7V001'])

    return df


def loadHomelessCounts(homelessCountsFile):

    print("Reading %s... " % homelessCountsFile, end='', flush=True)
    h = pd.read_csv(homelessCountsFile,dtype={'CensusCode':str})
    print("DONE.", flush=True)

    print("Found %i annual census tract subblocks in homeless counts data." % len(h.index))

    ## get the main census tract, stripping off sub-tract identifiers
    h['TRACTA'] = [x[0:6] for x in h.CensusCode]

    h.Unsheltered = h.Unsheltered.apply(lambda x: 0 if math.isnan(x) else x)
    h.Sheltered = h.Sheltered.apply(lambda x: 0 if math.isnan(x) else x)

    h['Total'] = h.Unsheltered + h.Sheltered

    h = h[['CensusCode','TRACTA','Name','Year','Unsheltered','Sheltered','Total']]

    x = h.groupby(['Year']).agg({
        'CensusCode':'count',
        'TRACTA':'nunique',
        'Unsheltered':'sum',
        'Sheltered':'sum',
        'Total':'sum'
        })
    print(x)

    ## summaries by main census tract
    hh = h.groupby(['TRACTA','Year']).sum()

    #x = hh.groupby(['Year']).agg({
    #    'Unsheltered':['count','sum'],
    #    'Sheltered':'sum',
    #    'Total':'sum'
    #    })
    #print(x)

    print("Found %i annual main census tract blocks in homeless counts data." % len(hh.index))

    ## create combined neighborhood names when main census tracts cross neigborhood boundaries
    hname = pd.DataFrame(h.groupby(['TRACTA'])['Name'].apply(lambda x: '/'.join(set(x))))


    ## merge in combined neighborhood names to main census tract stats
    hh = hh.merge(hname, left_index=True, right_index=True)

    #x = hh.groupby(['Year']).agg({
    #    'Unsheltered':['count','sum'],
    #    'Sheltered':'sum',
    #    'Total':'sum'
    #    })
    #print(x)

    return h, hh

def printDups(s):
    s = list(s)
    dups = {i:s.count(i) for i in s}
    dups2 = {i:v for i,v in dups.items() if v>1}
    print(dups2)


def calcHomelessStats(hcounts, tractShapes, tractPopulations):

    df = tractPopulations

    print("Calculating areas of each census tract... ",end='',flush=True)
    SQ_METERS_TO_SQ_MILES = 3.86102e-7
    areas = {f['properties']['TRACTCE']:area(f['geometry']) * SQ_METERS_TO_SQ_MILES for f in tractShapes['features']}
    print("DONE.")

    dfAreas = pd.DataFrame.from_dict(areas, orient = 'index', columns = ['area'])

#    print("Census tract shape duplicates")
#    printDups(dfAreas.index)


    # must use state code too, county codes are only unique within a state
    df = df[(df.STATEA == 6) & (df.COUNTYA == 37)]
    df = df[['TRACTA','H7V001']]
    df = df.set_index('TRACTA')
    df.columns = ['population']
    
    print("Found %i LA County census tract population blocks." % len(df.index))
#    print(df)

    df = df.merge(dfAreas, left_index=True, right_index=True, how='inner')
    df['density'] = df.population / df.area

#    print(df)

#    print("Census tract population duplicates")
#    printDups(df.index)

    hh = hcounts

    
    df = df.rename_axis('TRACTA')
    hh = hh.reset_index(level=['Year'])
#    print(hh)



    df = hh.join(df, how='inner')
    #df = df.merge(hh, left_on=['TRACTA'], right_on=['TRACTA'], how='inner')
    df['hdense'] = df.Unsheltered / df.area
    df['hrate'] = df.Unsheltered / df.population
    df['hconflict'] = (df.density * df.hdense) ** (1/2)

    x = hh.groupby(['Year']).agg({
        'Unsheltered':['count','sum'],
        'Sheltered':'sum',
        'Total':'sum'
        })
    print(x)


    return df

def setup_colorscale(df, chartVar):

    ## base year is 2016, future years should use same colorscale

    hsub = df[df.Year == 2016].drop(columns=['Year'])

    df = hsub

    ## the sequential colors are bad because there isn't enough separation at the high end

    #colors = px.colors.cmocean.thermal
    #colors = px.colors.sequential.Viridis
    #colors = px.colors.sequential.Inferno
    #colors = px.colors.sequential.YlOrRd

    # Picnic not good because the white midpoint does dot differentiate between areas that don't report data
    ##colors = px.colors.diverging.Picnic 

    # colors = px.colors.diverging.Geyser
    colors = px.colors.diverging.RdYlGn # need to reverse
    colors.reverse()

    max_data = max(hsub[chartVar])

    max_dense = 1000
#    max_dense = max_data
#    max_dense = log(max_data)

#    df = df.sort_values(by=chartVar)
#    bins = eqdist(df.Unsheltered, df[chartVar], len(colors)-1)

    #bins = [125 * x for x in range(0,len(colors))]
    #bins = [0,5,10,20,40,80,160,320,640,1280,2560,5120]
    #bins = [0,125,200,300,400,500,600,700,800,900,1000]
    
    
    incmult = 1.5
    start = max_dense / incmult**(len(colors)-2)
    bins = [-1] + [start * incmult**x for x in range(0,len(colors)-1)]

    #bb = np.linspace(0,1,len(colors))
    #hsub2 = hsub[hsub.hdense > 0]
    #bins = hsub2[chartVar].quantile(bb)

    #base = max_data ** (1/float(len(colors)-1))
    #bins = [0]
    #for i in range(1,len(colors)):
    #    bins.append(base**i)


#    print(bins)
    scale = list(np.divide(bins, max_dense))
    scale = np.minimum(1.0, scale)
    scale[0] = 0
    scale[len(colors)-1] = 1
#    print(scale)
    #scale = np.linspace(0,1,len(colors))
    #print(scale)

    #colorscale = px.colors.make_colorscale(colors, scale)
    colorscale = [list(tup) for tup in zip(scale, colors)]
#    print(colorscale)

    #exit(0)

    print(len(hsub.index))

    aggs = {
        'Name':'count',
        'Unsheltered':['sum','mean'],
        'Sheltered':['sum','mean'],
        'Total':['sum','mean'],
        'population':'sum',
        'hdense':'mean',
        'hrate':'mean',
        'hconflict':'mean'
        }
    pp = hsub.groupby(pd.cut(hsub[chartVar],bins)).agg(aggs)
    pptot = hsub.groupby(pd.cut(hsub[chartVar],[-1,50000])).agg(aggs) #.to_frame().assign(chartVar="Total").set_index(chartVar,append=True)
    pptot = pptot.reset_index()
    pptot[chartVar] = 'TOTAL'

    ppboth = pd.concat([pp.reset_index(),pptot],ignore_index=True).set_index(chartVar)
#    ppboth = pp.append(pptot)


    print("Colorscale binning results")
    print("--------------------------")
    print(ppboth)
#    print(pp)
#    print(pptot)



    #nonzero = df[df[chartVar] > 0]
    #qs = [0,0.001,0.005,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,0.991,0.992,0.993,0.994,0.995,0.996,0.997,0.998,0.999,1]
    #qq = nonzero[chartVar].quantile(qs)

    #pp = df.groupby(pd.cut(df.hrate,qq)).agg(['mean','count','sum'])
    #print(pp)

    return colorscale



def dumpTops(df):
    yy = df[df.Year == 2019]

    z = yy.sort_values(by='hconflict', ascending=False)
    print(z.head(25))

    z = yy.sort_values(by='hdense', ascending=False)
    print(z.head(25))

    z = yy.sort_values(by='hrate', ascending=False)
    z = z[z.population > 1000]
    print(z.head(25))

def mapPropertyToCensusTract(filename, **kwargs):

    prop_df = kwargs['prop_df']
    tract_df = kwargs['tract_df']

#    prop_df = prop_df.iloc[:200000]


    gg2x = tract_df[['TRACTCE','COUNTYFP','centroid','geometry']]


    glacprop = gp.GeoDataFrame(prop_df)
    # .set_index('AIN')
    def makePoints(filename, **kwargs):
        return gp.GeoSeries([sp.geometry.Point(p) for p in glacprop.coord])

    print("Creating shapely.Points for %i property records..." % len(glacprop.index))
    glacprop['centroid'] = fromPickOrParse(makePoints, picklefile='lacpropcentroids.pickle')



    def makeValid(filename, **kwargs):
        return [x.is_valid for x in glacprop.centroid]

    print("Determining valid points...")
    valid = fromPickOrParse(makeValid, picklefile='validcentroids.pickle')
    
    numvalid = valid.count(True)

    print("Keeping only %i valid points..." % numvalid)
    glacprop2 = glacprop[valid]

    glacprop2 = glacprop2[['AIN','centroid']]

    print("Finding container census tracts for each of %i properties..." % len(glacprop2.index))
    closest = near.get_containers(glacprop2, gg2x, k_neighbors=30)

    print(closest)

    closest = closest.drop(columns=['geometry','centroid_y','index_x','centroid_x','index_y'])

#    closest2 = closest[['AIN','COUNTYFP','TRACTCE']]
#    closest2.to_csv('junk66.csv')

    return closest

def loadPropertyData():

    gg2 = fromPickOrParse(url=GEOJSON_CALIFORNIA,
                        loadfunc=pd.read_pickle, 
                        dumpfunc=pd.to_pickle,
                        parsefunc=gp.read_file,
                        picklefile='_geojson.geopandas.pickle',
                        urlcachereadmode='rb')

    gg2['centroid'] = gg2.geometry.centroid

    counties = set([
        '037', # LA
        '111', # Ventura
        '059', # Orange
        '065', # Riverside
        '071', # San Bernardino
        '029'
        ])

    counties = set(['037'])

    gg2 = gg2[gg2.COUNTYFP.isin(counties)]
    print(gg2)


    lacprop = dfFromPickOrCSV(filename='Assessor_Parcels_Data_-_2018.csv')


    def addCoord(df):
        ### need this for later matching with other datasets, creating shapely points, etc
        df['coord'] = [(x,y) for x,y in zip(df.CENTER_LON, df.CENTER_LAT)]
        return df

    lacprop = addCoord(lacprop)

    propcats = [
        'PropertyType',
        'PropertyUseCode',
        'GeneralUseType',
        'SpecificUseType',
        'SpecificUseDetail1',
        'SpecificUseDetail2'
    ]

    aggstats = {'TotalValue':['count','sum','mean']}

    for cat in propcats:
        print(lacprop.groupby(by=cat).agg(aggstats))

    ca = lacprop.groupby(by=propcats).agg(aggstats)
    #ca.to_csv('propcatsfreq.csv')


    #.groupby(['AIN']).agg({'contains':'sum','distance':['min','mean','max']})






    closest = fromPickOrParse(parsefunc=mapPropertyToCensusTract, 
                              picklefile='closest3.pickle', 
                              prop_df=lacprop, 
                              tract_df=gg2)

    #closest =  closest.drop(columns=['index_x','centroid_x','index_y'])
    #closest.to_pickle('closest2.pickle')

    print(closest)

    closest2 = closest[['AIN','COUNTYFP','TRACTCE']]
    closest2 = closest2.set_index('AIN')

    lacprop2 = lacprop[['AIN','PropertyType','SpecificUseType','TotalValue','HouseNo','StreetName','City','ZIPcode5','coord']]
    lacprop2 = lacprop2.set_index('AIN')



    merged = lacprop2.merge(closest2, left_index=True, right_index=True, how='inner')

    prop_tract = merged.groupby(['TRACTCE','PropertyType']).agg({'TotalValue':['mean','sum','count']})

    print(prop_tract)
    #prop_tract.to_csv('prop_tract.csv')



    ### find out how many property types per location
    ### answer is, only 725 out of 2.4mm have more than 1 property type
    ###
    ### for this analysis, just delete these.
    ###
    ### better way would be to:
    ### just take the property type with the highest total value
    ### (usually the other type(s) are a mistake/zero value/oddity)

    def handleMultiTypeProperties(df):
        r = df.groupby(['coord']).PropertyType.nunique()

        rr = r.loc[r > 1]
        rr.rename('keep')

        df = df.merge(rr, on='coord', how='inner')

        return df

    lacprop2 = handleMultiTypeProperties(lacprop2)


    lacparcels = dfFromPickOrCSV(filename='LA_County_Parcels_no_BOM.csv', 
                                 usecols=['AIN','CENTER_LAT','CENTER_LON','ShapeSTArea','Roll_Year','Roll_LandValue','Roll_ImpValue']
                                 )

    lacparcels = addCoord(lacparcels)


    return None



def doChart(tracts, df, colorscale, chartVar, chartLabel, year = None): 

    constrain_range = False

    ## use these constraints if constrain_range == True
    zmin = 0
    zmax = df[chartVar].max()


    kwargs = {}
    if year:
        df = df[df.Year == year]
        if constrain_range:
            kwargs = dict(zmin = zmin, zmax = zmax)
    else:
        kwargs = dict(animation_frame='Year')
        if constrain_range:
            kwargs['range_color'] = (zmin, zmax)



    center = dict(lat=34.0522, lon=-118.2437)
    zoom = 10

    featureidkey = 'properties.TRACTCE'

    mapbox_style = 'carto-positron'

    colorscale = px.colors.diverging.RdYlGn_r # need to reverse

    opacity = 0.3

    z = np.log(np.maximum(0.01,df[chartVar]))

    if True:
        fig =  px.choropleth_mapbox(df, 
            geojson=tracts, 
            featureidkey=featureidkey,
            locations=df.index, 
            color=z,
            color_continuous_scale=colorscale,
            mapbox_style=mapbox_style,
            zoom=zoom, 
            center=center,
            opacity=opacity,
            labels=chartLabel,
            **kwargs
            )
    else:
        fig = go.Figure(go.Choroplethmapbox(geojson=tracts, 
                                            locations=df.index, 
                                            z=z,
                                            colorscale=colorscale,
                                            featureidkey=featureidkey,
                                            marker_opacity=opacity, 
                                            marker_line_width=0,
                                            **kwargs))

        fig.update_layout(mapbox_style=mapbox_style,
                          mapbox_zoom=zoom, 
                          mapbox_center=center
                          )

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


    #fig.write_image('junk.png')
    #cb = fig.data[0].colorbar


    #cb.tickmode = 'array'
    #cb.ticktext = ['0','10','100','1,000','10,000']
    #cb.tickvals = [0,10,100,1000,10000]

    tickrange = range(-4,8,2)
    fig.update_layout(
        coloraxis_colorbar=dict(
            tickmode='array',
            tickvals=list(tickrange),
            ticktext=[np.exp(x) for x in tickrange])
        )

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


#exit(0)

def dumpTractStats(df, outfile):
    df = df[['Name','Year','Unsheltered','Sheltered','Total','population','area','density','hdense','hrate','hconflict']]

    df.to_csv(outfile)


def doEverything():

    tracts = loadCensusTractShapes(GEOJSON_CALIFORNIA)

    pops = loadCensusTractPopulations(CENSUS_TRACT_POPULATIONS)

    hcountsSub, hcounts = loadHomelessCounts(HOMELESS_COUNTS)

    df = calcHomelessStats(hcounts, tracts, pops)

#    dumpTractStats(df, 'tractstats99.csv')

#    dumpTops(df)

#    prop_df = loadPropertyData()

    chartVar = 'hdense'
    chartLabel = {'Unsheltered Homeless Density (per sq mile)',chartVar}

    colorscale = setup_colorscale(df, chartVar)




    #doChart(df, chartVar, chartLabel)
    doChart(tracts, df, colorscale, chartVar, chartLabel, 2019)



if __name__ == '__main__':
    doEverything()

#gg = pd.DataFrame.from_dict(tracts, orient = 'index')
#gg = pd.json_normalize(tracts['features'])
#print(gg)
