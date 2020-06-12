::for %y in (2016 2017 2018 2019) do call jqproc.bat %y *%y*.json > %z.jqproc

for %y in (2016 2017 2018 2019) do for %z in (*%y*.json) do echo call jqproc.bat %y %z

::for %y in (2016 2017 2018 2019) do for %z in (*%y*.json) do call jqproc.bat %y %z > %z.jqproc
for %y in (2016 2017 2018 2019) do for %z in (*%y*.json) do call jqproc.bat %y %z

