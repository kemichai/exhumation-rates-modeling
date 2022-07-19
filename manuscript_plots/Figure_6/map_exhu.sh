#
# GMT code to plot seismicity and cross sections
# KM Sep 2017
#

out=Gcubed_fig6.eps

gmt set FORMAT_GEO_MAP D
gmt set FORMAT_GEO_MAP D
gmt set PS_MEDIA A0
gmt set FONT_ANNOT_PRIMARY Helvetica
gmt set FONT_ANNOT_PRIMARY 12
gmt set FONT_LABEL Helvetica
gmt set LABEL_FONT_SIZE 7
DEMdir="/home/kmichall/Desktop/topo"

#
# set -o nounset
# set -o errexit

DEMdir="/home/kmichall/Desktop/topo"

# Define map characteristics
# Define your area
north=-42.5
south=-44.5
east=171.8
west=168.65


proj='-JM6i'
gmt makecpt -Cviridis -T0/8/0.5 -Z  > seis.cpt
gmt makecpt -Cgray -Z -T0/5000/200 -I > topo.cpt


# echo Make basemap ...
# # make a basemap
echo Plotting coast...
gmt pscoast -W1/0.05 -Df $proj -R$west/$east/$south/$north -K -Y16 -B0.5wsEn -L169.25/-42.8/-42./50+l+u -P > $out


echo Using this clipped grid ....
#gmt grdimage -R -J $DEMdir/clipped_topo.grd -CFrance2.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out
#gmt grdimage -R -J $DEMdir/clipped_topo.grd -Ctopo.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out


echo Plotting lakes...
gmt psxy -R -J ../GMT_files/nz.gmt -W0.05,black -Gwhite -O -K >> $out
echo Plotting the coast...
gmt pscoast -W1/0.05 -Df -J -R -K -O -Swhite -P -L169.25/-42.8/-42./50+l+u >> $out

#####################################
echo Plotting cross-section lines...
start_lon='169.28'
start_lat='-44.0575'
end_lon='171.28'
end_lat='-43.0575'
width='2.5'
gmt psxy << END -R -J -O -W0.5,black,- -K>> $out
$start_lon $start_lat
$end_lon $end_lat
END
gmt pstext -R -J -D0/0.23 -O -K -F+f9p,Helvetica,gray10+jB -TO -Gwhite -W0.1 >> $out << END
$start_lon $start_lat A
END
gmt pstext -R -J -D0/0.23 -O -K -F+f9p,Helvetica,gray10+jB -TO -Gwhite -W0.1 >> $out << END
$end_lon $end_lat A'
END

start_lon_per='169.73'
start_lat_per='-43.57'
end_lon_per='170.33'
end_lat_per='-43.97'


width='2.5'
gmt psxy << END -R -J -O -W0.5,black,- -K>> $out
$start_lon_per $start_lat_per
$end_lon_per $end_lat_per
END
gmt pstext -R -J -D0/0.23 -O -K -F+f9p,Helvetica,gray10+jB -TO -Gwhite -W0.1 >> $out << END
$start_lon_per $start_lat_per B
END
gmt pstext -R -J -D0/0.23 -O -K -F+f9p,Helvetica,gray10+jB -TO -Gwhite -W0.1 >> $out << END
$end_lon_per $end_lat_per B'
END
#####################################


echo Plotting faults...
#gmt psxy -R -J ../GMT_files/activefaults.xy -Wlightred -W.8p -O -K >> $out
gmt psxy -R -J ../GMT_files/activefaults.xy -Wgray20 -W1p -O -K >> $out

gmt pstext -R$west/$east/$south/$north -J -O -K  -F+f10p,Helvetica,gray10+jBL+a32  >> $out << END
169.076 -43.876 Alpine Fault
# 170.857 -43.098 Alpine Fault
END
gmt pstext -R -J -O -K  -F+f10p,Helvetica,gray10+jBL+a0 -Gwhite >> $out << END
# 171.45 -42.7 Hope Fault
171.65 -42.7 HF
END
#Mount Cook
gmt psxy -R -J -Sx.3 -W1p -Gwhite -O -K  >> $out << END
170.1410417 -43.5957472
END

echo Plotting Lake names...
gmt pstext -R -J -O -K  -F+f8p,Helvetica,navy+jBL+a0 -Gwhite >> $out << END
170.5 -43.9 Lake
170.46 -43.95 Tekapo
170.12 -44.05 Lake
170.08 -44.1 Pukaki
169.71 -44.3 Lake
169.71 -44.35 Ohau
END

echo Plotting Toponyms labels...
gmt pstext -R -J -O -K -F+f9p,Helvetica,gray9+jB  >> $out << END
# gmt pstext -R -J -O -K -F+f12p,Times-Italic+jLM >> $out << END
170.175 -42.980 Harihari
170.7 -42.7 Hokitika
169.595 -43.3 Fox
169.60 -43.35 Glacier
169.79 -43.19 Franz Josef
169.79  -43.24 Glacier
170.63 -42.87 Ross
170.0 -43.10 Whataroa
168.83 -43.85 Haast
# 170.166667 -44.116667 Lake Pukaki
END
####################################################
#lines
####################################################
echo Plot lines that connect Toponyms to their labels...
#Harihari
gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
170.56 -43.15
170.28 -43.0
END
#fox
gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
170.017778 -43.464444
169.685 -43.371
END
# #Franz
gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
170.181944 -43.389167
169.890 -43.266
END
#whataroa
gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
170.11 -43.115
170.36 -43.262
END

####################################################
echo Plotting Toponyms as squares...
gmt psxy -R -J -Ss.1 -W1p -Gblack -O -K  >> $out << END
170.56 -43.15
170.96 -42.71
170.017778 -43.464444
170.181944  -43.389167
170.814167 -42.895833 #ross
170.36 -43.262   # whataroa
169.042222 -43.881111 #haast
END


gmt makecpt -Cviridis -T0/8/1 -Z  > seis.cpt
# Modelled
# awk '{print $1, $2 + 0.01, $3}' mod_uplifts.txt | gmt pstext -R -J -O -K -F+f6p,Helvetica,gray10+jB -Gwhite >> $out
awk '{print $1, $2, $3, 0.52}' Exhum_grid.txt | gmt psxy -i0,1,2,3s0.05 -Sc0.1 -R -J -O -K  -W.5 -Cseis.cpt >> $out
awk '{print $1, $2, $3, 3}' mod_uplifts_alpha1.txt | gmt psxy -i0,1,2,3s0.05 -Sd0.15 -R -J -O -K  -W.5 -Cseis.cpt >> $out

grid='-I9+k'
# grid='-I.1'
# gmt psxy uplift.dat -R -JX -Sc0.05i -Gblack -P -K -Y6.45i > $out
gmt blockmean mod_uplifts_alpha1.txt -R $grid > mean.xyz
gmt surface mean.xyz -R $grid -T0.6 -Gdata.nc
#gmt grdcontour data.nc -J -B -C1 -A2 -Gd5c -S0.1 -O -K -L0/10 -Wathin,black -Wcthinner,gray30 >> $out
#gmt grdcontour data.nc -J -B -Cseis.cpt -A1 -Gd5c -S0.1 -O -K -L0/10 -Wathin+c -Wcthinner+c >> $out
gmt grdcontour data.nc -J -B -Cseis.cpt -A1 -Gd5c -S10  -O -K -L0/10 -Wathick+c -Wcthickest+c >> $out

# -A is annotation interval in data units
# -L sets the limits
# -GDk5 every five km

gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt  -Bx2f1 -By+l"Exhumation rate (mm/yr)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out
# gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt  -Bx10f5 -By+l" Hypocentral depths (km)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out



echo Plotting GPS stations...
awk '{print $3, $2}' ../GMT_files/GPS_sta_GEONET.txt |
    gmt psxy -R -J -Sd.2 -W0.4p -Gdarkorange2 -O -K  >> $out





echo Plot scale ...
# gmt psscale -Dx1/9.5+o0/0.6i+w1.5i/0.08i+h+e -R -J -CFrance2.cpt -Bx2000f1000 -By+l"Topography (m)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out

# gmt set FONT_ANNOT_PRIMARY 9

echo Creating legend...
# # construct legend
gmt pslegend <<END -R -J -Dx4.65i/0.1i+w1.2i/1.1i/TC -C0.1i/0.1i -F+gwhite+pthin -P -O -K >> $out
G -.01i
S .04i c .1i white 0.2p 0.18i Grid points
G .07i
# S .04i + .11i black 0.2p 0.18i TC data
# G .07i
S .04i d .11i darkorange2 0.2p 0.18i GPS
G .07i
S .04i s .08i black 0.2p 0.18i Towns
G .065i
S .04i - .14i red thick 0.18i Active fault
END






#######################################
# Second map
#######################################

echo Plotting coast...
gmt pscoast -W1/0.05 -Df $proj -R$west/$east/$south/$north -K -O -Y-14 -Ggrey -B0.5wSEn -L169.25/-42.8/-42./50+l+u -P >> $out

echo Using this clipped grid ....
# gmt grdimage -R -J $DEMdir/clipped_topo.grd -CFrance2.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out
#gmt grdimage -R -J $DEMdir/clipped_topo.grd -Ctopo.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out


echo Plotting lakes...
gmt psxy -R -J ../GMT_files/nz.gmt -W0.05,black -Gwhite -O -K >> $out
echo Plotting the coast...
gmt pscoast -W1/0.05 -Df -J -R -K -O -Swhite -P -L169.25/-42.8/-42./50+l+u >> $out

echo Plotting faults...
#gmt psxy -R -J ../GMT_files/activefaults.xy -Wlightred -W.8p -O -K >> $out
gmt psxy -R -J ../GMT_files/activefaults.xy -Wgray20 -W1p -O -K >> $out

gmt pstext -R$west/$east/$south/$north -J -O -K  -F+f10p,Helvetica,gray10+jBL+a32  >> $out << END
169.076 -43.876 Alpine Fault
# 170.857 -43.098 Alpine Fault
END
gmt pstext -R -J -O -K  -F+f10p,Helvetica,gray10+jBL+a0 -Gwhite >> $out << END
# 171.45 -42.7 Hope Fault
171.65 -42.7 HF
END
#Mount Cook
gmt psxy -R -J -Sx.3 -W1p -Gwhite -O -K  >> $out << END
170.1410417 -43.5957472
END

echo Plotting Lake names...
gmt pstext -R -J -O -K  -F+f8p,Helvetica,navy+jBL+a0 -Gwhite >> $out << END
170.5 -43.9 Lake
170.46 -43.95 Tekapo
170.12 -44.05 Lake
170.08 -44.1 Pukaki
169.71 -44.3 Lake
169.71 -44.35 Ohau
END

# echo Plotting Toponyms labels...
# gmt pstext -R -J -O -K -F+f9p,Helvetica,gray9+jB  >> $out << END
# # gmt pstext -R -J -O -K -F+f12p,Times-Italic+jLM >> $out << END
# 170.175 -42.980 Harihari
# 170.7 -42.7 Hokitika
# 169.595 -43.3 Fox
# 169.60 -43.35 Glacier
# 169.79 -43.19 Franz Josef
# 169.79  -43.24 Glacier
# 170.63 -42.87 Ross
# 170.0 -43.10 Whataroa
# 168.83 -43.85 Haast
# # 170.166667 -44.116667 Lake Pukaki
# END
####################################################
#lines
####################################################
# echo Plot lines that connect Toponyms to their labels...
# #Harihari
# gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
# 170.56 -43.15
# 170.28 -43.0
# END
# #fox
# gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
# 170.017778 -43.464444
# 169.685 -43.371
# END
# # #Franz
# gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
# 170.181944 -43.389167
# 169.890 -43.266
# END
# #whataroa
# gmt psxy -R -J -Wblack -W0.5p -O -K  >> $out << END
# 170.11 -43.115
# 170.36 -43.262
# END

####################################################
echo Plotting Toponyms as squares...
gmt psxy -R -J -Ss.1 -W1p -Gblack -O -K  >> $out << END
170.56 -43.15
170.96 -42.71
170.017778 -43.464444
170.181944  -43.389167
170.814167 -42.895833 #ross
170.36 -43.262   # whataroa
169.042222 -43.881111 #haast
END

# Modelled
awk '{print $1, $2, $3, 3}' mod_uplifts_init.txt | gmt psxy -i0,1,2,3s0.05 -Sc0.15 -R -J -O -K  -W.5 -Cseis.cpt >> $out
# awk '{print $1, $2 + 0.01, $3}' mod_uplifts.txt | gmt pstext -R -J -O -K -F+f6p,Helvetica,gray10+jB -Gwhite >> $out
gmt makecpt -Cviridis -T0/8/1 -Z  > seis.cpt


grid='-I9+k'
# grid='-I.1'
# gmt psxy uplift.dat -R -JX -Sc0.05i -Gblack -P -K -Y6.45i > $out
gmt blockmean mod_uplifts_init.txt -R $grid > mean.xyz
gmt surface mean.xyz -R $grid -T0.6 -Gdata.nc
#gmt grdcontour data.nc -J -B -C1 -A2 -Gd5c -S0.1 -O -K -L0/10 -Wathin,black -Wcthinner,gray30 >> $out
# -A is annotation interval in data units
# -L sets the limits
# -GDk5 every five km
#gmt grdcontour data.nc -J -B -Cseis.cpt -A1 -Gd5c -S0.1 -O -K -L0/10 -Wathin+c -Wcthinner+c >> $out
#+cl for numbers to be black and contour lines colored
#gmt grdcontour data.nc -J -B -Cseis.cpt -A1 -Gd5c -S5 -O -K -L0/10 -Wathick+cl  >> $out
gmt grdcontour data.nc -J -B -Cseis.cpt -A1 -Gd5c -S10 -O -K -L0/10 -Wathick+c  >> $out

gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt  -Bx2f1 -By+l"Exhumation rate (mm/yr)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out
# gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt  -Bx10f5 -By+l" Hypocentral depths (km)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out



echo Plotting GPS stations...
awk '{print $3, $2}' ../GMT_files/GPS_sta_GEONET.txt |
    gmt psxy -R -J -Sd.2 -W0.4p -Gdarkorange2 -O -K  >> $out





echo Plot scale ...
# gmt psscale -Dx1/9.5+o0/0.6i+w1.5i/0.08i+h+e -R -J -CFrance2.cpt -Bx2000f1000 -By+l"Topography (m)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out

# gmt set FONT_ANNOT_PRIMARY 9

# echo Creating legend...
# # # construct legend
# gmt pslegend <<END -R -J -Dx5.02i/0.06i+w0.94i/0.9i/TC -C0.1i/0.1i -F+gwhite+pthin -P -O -K >> $out
# G -.01i
# S .04i c .05i white 0.2p 0.18i Grid points
# G .07i
# # S .04i + .11i black 0.2p 0.18i TC data
# # G .07i
# S .04i d .11i darkorange2 0.2p 0.18i GPS
# G .07i
# S .04i s .08i black 0.2p 0.18i Towns
# G .065i
# S .04i - .14i red thick 0.18i Active fault
# END



#rm -f mean.xyz track *.nc *.d gmt.conf

gmt psxy -R -J -T -O >> $out
gmt psconvert -Tf -A $out
#ps2raster -Tf -A map.ps
evince ${out%.*}.pdf




































out=Gcubed_fig9.eps

#

gmt set FORMAT_GEO_MAP D
gmt set PS_MEDIA A0
gmt set FONT_ANNOT_PRIMARY Helvetica
gmt set FONT_ANNOT_PRIMARY 18
gmt set FONT_LABEL Helvetica
gmt set LABEL_FONT_SIZE 20

gmt psbasemap -R0/6.5/0/7.5 -JX45/20 -P -B -K > $out




start_lon='169.28'
start_lat='-44.0575'
end_lon='171.28'
end_lat='-43.0575'
width='8'

# gmt grdcut $DEMdir/clipped_topo.grd -R169.28E/171.28E/44.0575S/43.0575S -Gspac_33.nc

# cat << EOF > ridge.txt
# 169.28 -44.0575
# 171.28 -43.0575
# EOF

# gmt grdtrack ridge.txt -G@spac_33.nc -C200k/1k/1k+v -Sa+sstack.txt > table.txt
# gmt convert stack.txt -o0,5 > env.txt
# gmt convert stack.txt -o0,6 -I -T >> env.txt

#Topography...
rm profile.xy
## Get the coordinates of points of 0.1 degree apart along the great circle arc from two points:
gmt sample1d -I0.01 << END >> profile.xy
$start_lon $start_lat
$end_lon $end_lat
END

# gmt grdtrack profile.xy -G$DEMdir/clipped_topo.grd  > profile.xyz

awk '{print($1, $2, $3)}' profile.xyz | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-1/1 -Q -Fpz > profile.dat


gmt psxy -R -J -O -W0.5,black,- -K >> $out << END
$start_lon $start_lat
$end_lon $end_lat
END

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
#
awk '{ print $3, $2, $4 ,$17}' ../GMT_files/hypoDD.reloc3 | gmt project -C$start_lon/$start_lat \
-E$end_lon/$end_lat -W-$width/$width -Q -Fpz > projection_a.dat

awk '{print($1, $2, -2)}' ../GMT_files/Aoraki.dat | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat \
-W-$width/$width -Q -Fpz > Aoraki_g.dat
# LFEs
awk '{print($1, $2, $3)}' ../GMT_files/LFE_LMB.txt | gmt project -C$start_lon/$start_lat -E$end_lon/$end_lat\
 -W-$width/$width -Q -Fpz > LFE_proj.dat




awk '{print($1,$2,$3)}' projection_a.dat | gmt psxy -Sci -i0,1,2s0.045 -W.25 -Gdimgrey \
-R0/190/-3/30 -JX30/-10 -Bx20+l"Distance (km)"  -By5+l"Depth (km)" -BwSnE -Y0 -X5 -O -K >> $out
awk '{print($1,$2,1)}' projection_100.dat | gmt psxy -W1.5,red,- \
-R -J -B -O -K >> $out
awk '{print($1,$2,1)}' projection_200.dat | gmt psxy -W1.5,red,- \
-R -J -B -O -K >> $out
awk '{print($1,$2,1)}' projection_300.dat | gmt psxy -W1.5,red,- \
-R -J -B -O -K >> $out
awk '{print($1,$2,1)}' projection_400.dat | gmt psxy -W1.5,red,- \
-R -J -B -O -K >> $out
awk '{print($1,$2,1)}' projection_500.dat | gmt psxy -W1.5,red,- \
-R -J -B -O -K >> $out

# awk '{print($1,$2,1)}' projection_600.dat | gmt psxy -W0.5,black \
# -R -J -B -O -K >> $out
#
awk '{print($1,-$2/1000)}' profile.dat | gmt psxy -W.25 \
-R -J -B -K -O >> $out
awk '{print($1,$2,$2)}' LFE_proj.dat | gmt psxy -Sa0.4  -W.25 -Gblack -R -J -O -K -V >> $out
awk '{print($1,$2,$2)}' Aoraki_g.dat| gmt psxy -Sx0.5  -W.5 -Gblack -R \
-J  -O -K -V >> $out

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


rm -f z.cpt ridge.txt table.txt env.txt #stack.txt


gmt psxy -R -J -T -O >> $out
gmt psconvert -Tf -A $out
#ps2raster -Tf -A map.ps
evince ${out%.*}.pdf














