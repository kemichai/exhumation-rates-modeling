out=Gcubed_fig9.eps
##############################
# Cross section along the length of the Alpine Fault
# of seismicity, modelled temperature isotherms
# LFEs and GPS locking depths.
# Konstantinos Michailos
# Modified: June 2020

gmt set FORMAT_GEO_MAP D
gmt set PS_MEDIA A0
gmt set FONT_ANNOT_PRIMARY Helvetica
gmt set FONT_ANNOT_PRIMARY 18
gmt set FONT_LABEL Helvetica
gmt set LABEL_FONT_SIZE 20


gmt psbasemap -R0/6.5/0/7.5 -B -JX45/20 -P -K > $out
# cross section coordinates
start_lon='169.28'   # A
start_lat='-44.0575' # A
end_lon='171.28'     # A'
end_lat='-43.0575'   # A'
# swath on each side of the cross section
width='10'



# create topography
rm profile.xy
# Get the coordinates of points of 0.1 degree
# apart along the great circle arc from two points:
gmt sample1d -I0.01 << END >> profile.xy
$start_lon $start_lat
$end_lon $end_lat
END
awk '{print($1, $2, $3)}' profile.xyz | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-1/1 -Q -Fpz > topo.dat


awk '{ print $1, $2, $3 }' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_100.dat
awk '{ print $1, $2, $4 }' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_200.dat
awk '{ print $1, $2, $5 }' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_300.dat
awk '{ print $1, $2, $6 }' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_400.dat
awk '{ print $1, $2, $7}' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_500.dat
awk '{ print $1, $2, $8 }' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_600.dat
awk '{ print $1, $2, $9 }' temps_par.txt | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > exh.txt
# project earthquake hypocenters along the cross section
awk '{ print $3, $2, $4 ,$17}' ../GMT_files/hypoDD.reloc3 | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > seismicity.dat
# Aoraki/Mount Cook
awk '{print($1, $2, -2)}' ../GMT_files/Aoraki.dat | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat \
-W-$width/$width -Q -Fpz > Aoraki_g.dat
# LFEs
awk '{print($1, $2, $3)}' ../GMT_files/LFE_LMB.txt | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat\
 -W-$width/$width -Q -Fpz > LFE_proj.dat
awk '{print($1, $2, $3)}' Wallace.txt | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat\
 -W-50/50 -Q -Fpz > GEOD_proj.dat
awk '{print($1, $2, $3)}' Lamb.txt | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat\
 -W-50/50 -Q -Fpz > GEOD2_proj.dat


# plot
awk '{print($1,$2,$3)}' seismicity.dat | gmt psxy -Sci -i0,1,2s0.045 -W.25 -Gdimgrey \
-R0/190/-3/30 -JX30/-10 -Bx20+l"Distance (km)" -BwSnE -By5+l"Depth (km)" -Y0 -X5 -O -K >> $out
awk '{print($1,$2,1)}' projection_100.dat | gmt psxy -W1.7,red,- -R -J -O -K >> $out
awk '{print($1,$2,1)}' projection_200.dat | gmt psxy -W1.7,red,- -R -J -O -K >> $out
awk '{print($1,$2,1)}' projection_300.dat | gmt psxy -W1.7,red,- -R -J -O -K >> $out
awk '{print($1,$2,1)}' projection_400.dat | gmt psxy -W1.7,red,- -R -J -O -K >> $out
awk '{print($1,$2,1)}' projection_500.dat | gmt psxy -W1.7,red,- -R -J -O -K >> $out
# awk '{print($1,$2,1)}' projection_600.dat | gmt psxy -W0.5,black -R -J -B -O -K >> $out
#
awk '{print($1,-$2/1000)}' topo.dat | gmt psxy -W.25 -R -J -K -O >> $out
awk '{print($1,$2,$2)}' LFE_proj.dat | gmt psxy -Sa0.4  -W.25 -Gblack -R -J -O -K -V >> $out
awk '{print($1,$2,$2)}' Aoraki_g.dat| gmt psxy -Sx0.5  -W1.5 -Gblack -R -J -O -K -V >> $out
awk '{print($1,$2,$2)}' GEOD_proj.dat | gmt psxy -Ss0.2  -W.25 -Ggreen -R -J -O -K -V >> $out
awk '{print($1,$2,$2)}' GEOD2_proj.dat | gmt psxy -Ss0.5  -W.25 -Gmediumpurple -R -J -O -K -V >> $out

# project cross
gmt pstext -R -JX -O -K -F+f18p,Helvetica,gray10+jB  -TO -Gwhite -W0.1 >> $out << END
3.5 0 A
185 0 A'
END

gmt pstext -R -JX -O -K -F+f12p,Helvetica,gray10+jB  -TO -Gwhite >> $out << END
5 2.1 100
5 4  200
5 6.5  300
5 9.7  400
5 13.5  500
END


# gmt psxy -R-200/200/0/5000 -Bxafg1000+l"Distance from ridge (km)" -Byaf+l"Depth (m)" -BWSne \
#  	-JX26i/3i -O -K -Glightgray env.txt -Y10.5i >> $out
# gmt psxy -R -J -O -K -W3p stack.txt >> $out

# awk '{print($1,$2,$3)}' projection_a.dat | gmt psxy -Sci -i0,1,2s0.045 -W.25 -Gdimgrey \
# -R0/190/-3/30 -JX30/-10 -Bx20+l"Distance (km)"  -By5+l"Depth (km)" -BwSnE -Y4 -X5 -O -K >> $out

awk '{ print $1, $2, 1 }' exh.txt | gmt psxy -W1.5,black,- \
-R0/190/0/10 -JX30/5 -Bx -By2+l"Exh. rate (mm/yr)" -BwSnE -O -K -Y11 >> $out

gmt pstext -R -JX -O -K -F+f18p,Helvetica,gray10+jB -TO -G  >> $out << END
3.5 8 SW
182 8 NE
END

rm -f z.cpt ridge.txt table.txt env.txt #stack.txt


gmt psxy -R -J -T -O >> $out
gmt psconvert -Tf -A $out
#ps2raster -Tf -A map.ps
evince ${out%.*}.pdf
