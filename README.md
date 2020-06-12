# **LA County Homeless Count Analysis**

### Retrieval of bulk LA County homeless count data using undocumented APIs, cleanup, and choropleth density charting.

This is an extremely raw hack. The retrieval and cleanup code programs are some of my very first Python programming attempts, so they a very poor example of my coding style. They do does the job, but that's about it.

The charting program is a little better in terms of "Pythonic" usage, but again not meant for public consumption. It's more meant to mess around with creating choropleths at the census tract level.

The workflow is this:

1) Get bulk JSON data using lahsa_counts_v2.py. Right now these are broken (they changed their API since I first wrote the code). I downloaded the latest data by hand by running a debugger inside Chrome. The API hook is https://wabi-west-us-api.analysis.windows.net/public/reports/querydata, but you need some API & resource keys (get by debugging in Chrome) and some structured query. The query setup is in the Python code but needs updating to the latest undocumented API.

2) Extract key elements from the JSON using jq (you can download that here: https://stedolan.github.io/jq/). It is called from the jqprocALL.bat file (which uses 4NT (Take Command) syntax... may not work with Windows cmd or Powershell... definitely not bash).

3) Cleanup the JSON extract using lahsa_clean_json_v1a.py. This part normalizes and cleans the JSON, which has unnamed elements with differing array lengths in each census tract. There are several assumptions made about how to fill missing data that may or may not be correct. However, I do not estimate any homeless counts if I do not find them. The output are CSV files.

4) Concatenate the yearly CSV files into one timeseries file using cat_all.bat.

5) Create a single year choropleth (or animated version across years) using homelesscharts.py. This part also depends on a regular population census count by census tract within California, which I downloaded from another source by hand. It also needs GEOJSON descriptions of each census tract in California, which are downloaded automatically via this program. This program does a lot of density distribution analysis to try to get a good density color scale on the map. That part is still a work in progress.

