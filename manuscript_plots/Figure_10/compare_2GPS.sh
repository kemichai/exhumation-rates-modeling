out=Gcubed_figure10.eps



#

gmt set FORMAT_GEO_MAP D
gmt set PS_MEDIA A0
gmt set FONT_ANNOT_PRIMARY Helvetica
gmt set FONT_ANNOT_PRIMARY 18
gmt set FONT_LABEL Helvetica
gmt set LABEL_FONT_SIZE 20

gmt psbasemap -R0/6.5/0/7.5 -JX -P -B -K > $out



# end_lon='170.4649'
# end_lat='-43.9857'
end_lon='170.33'
end_lat='-43.97'
start_lon='169.8158'
start_lat='-43.5317'
width='20'

# gmt grdcut $DEMdir/clipped_topo.grd -R169.28E/171.28E/44.0575S/43.0575S -Gspac_33.nc

# cat << EOF > ridge.txt
# 169.28 -44.0575
# 171.28 -43.0575
# EOF

# gmt grdtrack ridge.txt -G@spac_33.nc -C200k/1k/1k+v -Sa+sstack.txt > table.txt
# gmt convert stack.txt -o0,5 > env.txt
# gmt convert stack.txt -o0,6 -I -T >> env.txt

start_lon_per='169.73'
start_lat_per='-43.57'
end_lon_per='170.33'
end_lat_per='-43.97'

#Topography...
## Get the coordinates of points of 0.1 degree apart along the great circle arc from two points:
gmt sample1d -I0.01 << END >> profile.xy
$start_lon_per $start_lat_per
$end_lon_per $end_lat_per
END

# gmt grdtrack profile.xy -G$DEMdir/clipped_topo.grd  > profile.xyz

awk '{print($1, $2, $3)}' profile.xyz | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-1/1 -Q -Fpz > profile.dat


gmt psxy -R -J -O -W0.5,black,- -K >> $out << END
$start_lon $start_lat
$end_lon $end_lat
END


awk '{print($3, $2, $5, $6)}' ../GMT_files/GPS_sta_GEONET.txt | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat\
 -W-$width/$width -Q -Fpz > beavan.dat
awk '{print($3, $2, $5,$1)}' ../GMT_files/GPS_sta_GEONET.txt | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat\
 -W-$width/$width -Q -Fpz > beavan_names.dat
awk '{ print $1, $2, $9 }' temps_per.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-50/50 -Q -Fpz > exh_per.txt


awk '{ print $1, $2, $3 }' beavan.dat | gmt psxy -W1,red,- \
-R-20/80/0/8 -JX20/15 -Bx20f10+l"Distance (km)" -By2f1+l"Vertical. rate (mm/yr)" -BwSnE -Ey -Gred -O -K >> $out
awk '{ print $1, $2, 2 }' beavan.dat | gmt psxy -Sci -i0,1,2s0.05 -W.25 -Gred \
-R -J -BwSnE -O -K  >> $out
awk '{ print $1 + 0.2, $2 +.5 ,$3 }' beavan_names.dat |
gmt pstext -R -JX -O -K -F+f12p,Helvetica,gray10+jB+a90   >> $out


# +3
gmt psxy -W1,red,- -R -J  -Ey -Gred -O -K  >> $out << END
-22.8 0.7 0.2
6.9  1.9 0.4
19.3 4.8 0.9
20.6 4.0 0.2
25.7 4.8 0.3
55.0 0.7 0.5
END

gmt psxy -Sci -i0,1,2s0.05 -W.25 -Gred -R -J -BwSnE -O -K  >> $out << END
-22.8 0.7 2
6.9  1.9 2
19.3 4.8 2
20.6 4.0 2
25.7 4.8 2
55.0 0.7 2
END

gmt pstext -R -JX -O -K -F+f12p,Helvetica,gray10+jB+a90 >> $out << END
-22.8 0.7 HOKI (off transect)
6.9  1.9 LEOC
19.3 4.8 MAKA
20.6 4.0 REDD
25.7 4.8 MCKE
55.0 0.7 MTCX
END

awk '{ print $1, $2, 1 }' exh_per.txt | gmt psxy -W2,grey,- \
-R -J -O -K  >> $out


rm -f z.cpt ridge.txt table.txt env.txt #stack.txt


gmt psxy -R -J -T -O >> $out
gmt psconvert -Tf -A $out
#ps2raster -Tf -A map.ps
evince ${out%.*}.pdf
