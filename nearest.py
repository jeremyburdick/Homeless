### borrowed from https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html


from sklearn.neighbors import BallTree
import numpy as np
import pandas as pd
import multiprocessing as mp
from collections import namedtuple


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
#    closest = indices[0]
#    closest_dist = distances[0]
    closest = indices
    closest_dist = distances

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, left_geom_col = None, right_geom_col = None, return_dist=False, k_neighbors=1):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_geom_col or left_gdf.geometry.name
    right_geom_col = right_geom_col or right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians, k_neighbors=k_neighbors)

    closest_chunks = []

    for k in range(0, k_neighbors):
        # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
        chunk = right.loc[closest[k]]
        chunk['closest'] = k

        # Ensure that the index corresponds the one in left_gdf
        chunk = chunk.reset_index(drop=True)

        # Add distance if requested
        if return_dist:
            # Convert to meters from radians
            earth_radius = 6371000  # meters
            chunk['distance'] = dist[k] * earth_radius

        chunk = left_gdf.reset_index().merge(chunk.reset_index(), left_index=True, right_index=True, how='left')
#        chunk = left_gdf.merge(chunk)

        closest_chunks.append(chunk)
        print(chunk)

    closest_all = pd.concat(closest_chunks)

    return closest_all




def worker(ichunk, tree, left_points, right_polys, k_neighbors, results):

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(left_points.apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    print("Radians %i. DONE." % ichunk, flush=True)

    # Find the nearest points
    # -----------------------
    # distances ==> distance between the nearest neighbors (in meters)
    # indices ==> index in right_gdf that corresponds to the closest point

    distances, indices = tree.query(left_radians, k=k_neighbors)

    print("Nearest %i. DONE." % ichunk, flush=True)

    centroids = list(left_points)
    polys = list(right_polys)

    ## initial "container" is empty (last intentionally blank row index in census tract centers)
    contains = [len(polys)] * len(indices)

    for i, near in enumerate(indices, 0):
#        if i % 5000 == 0: print(end=".", flush=True)

        cent = centroids[i]
        for k in near:
            poly = polys[k]
            if poly.contains(cent):
                contains[i] = k
                break

    print("Contains %i. DONE." % ichunk, flush=True)

    ## copy results
    for i in range(0,len(contains)):
        results[i] = contains[i]



def get_containers(left_gdf, right_gdf, left_points_col = 'centroid', right_points_col = 'centroid', right_polygons_col = 'geometry', return_dist=False, k_neighbors=50):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)


    print("Converting %i candidate points to radians..." % len(right.index))
    right_radians = np.array(right[right_points_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    print("Making BallTree with %i candidate points..." % len(right_radians))
    # Create tree from the candidate points
    tree = BallTree(right_radians, leaf_size=15, metric='haversine')


    ## setup a blank record for "not found" results
    blank = right.iloc[0]
    for x in range(0, len(blank.index)):
        blank.iat[x] = None

    ## append the "not found" row
    right = pd.concat([right,blank]).reset_index()





    nrows = len(left_gdf.index)
    nchunks = mp.cpu_count()
    chunksize = int(nrows / nchunks) + (1 if nrows % nchunks > 0 else 0)

    left_gdf_orig = left_gdf
        

    pp = []

    JobChunk = namedtuple('JobChunk',['process','left_slice','results','ichunk'])

    ## spawn workers
    for i in range(0, nchunks):

        ichunk = 1+i

        print("Slicing chunk %i." % ichunk)

        slicestart = i*chunksize
        sliceend = min(nrows,(i+1)*chunksize)
        thischunksize = sliceend - slicestart

        sl = slice(slicestart,sliceend)
        left_gdf = left_gdf_orig.iloc[sl]
        
        results = mp.Array('i',[-1] * thischunksize)

        p = mp.Process(target=worker, args=(1+i, tree, left_gdf.centroid, right_gdf.geometry, k_neighbors, results,))

        print("Launching chunk %i." % ichunk)
        p.start()

        pp.append(JobChunk(p, left_gdf, results, ichunk))

    ## wait for results
    for jc in pp:
        jc.process.join()
        print("Finished %i." % jc.ichunk)

    chunks = []

    for jc in pp:

        contains = jc.results[:]
        chunk = right.iloc[contains]
        chunk = chunk.reset_index(drop=True)

        chunk = jc.left_slice.reset_index().merge(chunk, left_index=True, right_index=True, how='left')

        chunks.append(chunk)

        print("DONE. %i." % jc.ichunk, flush=True)

    chunks2 = pd.concat(chunks)

    return chunks2



