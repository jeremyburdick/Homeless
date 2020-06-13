
set y={y%1:.}

cat %2 | jq ".results[].result.data.dsr.DS[0].PH[0].DM0[].C" | jq -s "%y" | jq -s "{data:.}" 

