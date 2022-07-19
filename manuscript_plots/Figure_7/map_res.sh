#
# GMT code to plot seismicity and cross sections
# KM Sep 2017
#

out=Gcubed_figure7.eps

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
gmt grdimage -R -J $DEMdir/clipped_topo.grd -Ctopo.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out


echo Plotting lakes...
gmt psxy -R -J ../GMT_files/nz.gmt -W0.05,black -Gwhite -O -K >> $out
echo Plotting the coast...
gmt pscoast -W1/0.05 -Df -J -R -K -O -Swhite -P -L169.25/-42.8/-42./50+l+u >> $out

gmt makecpt -Cpolar -T-20/20/0.1 -D+i > seis.cpt

grid='-I7+k'
# grid='-I.1'
# gmt psxy uplift.dat -R -JX -Sc0.05i -Gblack -P -K -Y6.45i > $out
gmt blockmean res_alpha1.dat -R $grid > res_mean.xyz
# gmt psmask res_mean.xyz -J -I30+k -R$west/$east/$south/$north  -O -K  >> $out
gmt surface res_mean.xyz -R $grid -T0.7 -Gdata.nc
# gmt grdcontour data.nc -J -B -C1 -A2 -Gd10c -S0.1 -O -K -L0/10 -Wathick,black -Wcthinnest,gray30,- >> $out
gmt grdview data.nc -R -J -B -Cseis.cpt -Qs -O -K  >> $out

#The name of the CPT. Must be present if you want (1) mesh plot with contours (-Qm),
# or (2) shaded/colored perspective image (-Qs or -Qi). For -Qs: You can specify that
#  you want to skip a z-slice by setting the red r/g/b component to -; to use a pattern give
#   red = P|ppattern[+bcolor][+fcolor][+rdpi]. Alternatively, supply the name of a GMT
#    color master dynamic CPT [rainbow] to automatically dete
#rmine a continuous CPT from the grid’s z-range; you may round up/down the z-range by adding +izinc..
# -A is annotation interval in data units
# -L sets the limits
# -GDk5 every five km
awk '{print $1, $2, $3, 3}' thermo_res_alpha1.csv | gmt psxy -i0,1,2,3s0.05 -S+0.15 -R -J -O -K  -W.8  >> $out
awk '{print $1, $2, $3, 3}' seis_res_alpha1.csv | gmt psxy -i0,1,2,3s0.05 -Sd0.15 -R -J -O -K  -W.8 >> $out
gmt pscoast -W1/0.05 -Swhite -Df -J -R -K -O  -L169.25/-42.8/-42./50+l+u >> $out

gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt -Bx10f5 -By+l"Residual" -O -K  --FONT_ANNOT_PRIMARY=10p >> $out
#
awk '{print $1, $2}' mod_uplifts_alpha1.txt | gmt psxy -i0,1 -Sc0.15 -R -J -O -K  -W.5 -Gwhite >> $out


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







echo Plot scale ...
# gmt psscale -Dx1/9.5+o0/0.6i+w1.5i/0.08i+h+e -R -J -CFrance2.cpt -Bx2000f1000 -By+l"Topography (m)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out

# gmt set FONT_ANNOT_PRIMARY 9

echo Creating legend...
# # construct legend
gmt pslegend <<END -R -J -Dx4.55i/0.1i+w1.35i/1.1i/TC -C0.1i/0.1i -F+gwhite+pthin -P -O -K >> $out
G -.01i
S .04i d .05i white 0.2p 0.18i Seis. data
G .07i
S .04i + .11i black 0.2p 0.18i Thermochron.
G .07i
S .04i s .08i black 0.2p 0.18i Towns
G .065i
S .04i - .14i red thick 0.18i Active fault
END




#######################################
# Second map
#######################################

echo Plotting coast...
gmt pscoast -W1/0.05 -Df $proj -R$west/$east/$south/$north -K -O -Y-14 -B0.5wSEn -L169.25/-42.8/-42./50+l+u -P >> $out

echo Using this clipped grid ....
# gmt grdimage -R -J $DEMdir/clipped_topo.grd -CFrance2.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out
gmt grdimage -R -J $DEMdir/clipped_topo.grd -Ctopo.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out






echo Plotting lakes...
gmt psxy -R -J ../GMT_files/nz.gmt -W0.05,black -Gwhite -O -K >> $out
echo Plotting the coast...
gmt pscoast -W1/0.05 -Df -J -R -K -O -Swhite -P -L169.25/-42.8/-42./50+l+u >> $out


grid='-I7+k'
# grid='-I.1'
# gmt psxy uplift.dat -R -JX -Sc0.05i -Gblack -P -K -Y6.45i > $out
gmt blockmean res_init.dat -R $grid > res_mean.xyz
# gmt psmask res_mean.xyz -J -I30+k -R$west/$east/$south/$north  -O -K  >> $out
gmt surface res_mean.xyz -R $grid -T0.7 -Gdata.nc
# gmt grdcontour data.nc -J -B -C1 -A2 -Gd10c -S0.1 -O -K -L0/10 -Wathick,black -Wcthinnest,gray30,- >> $out
gmt grdview data.nc -R -J -B -Cseis.cpt -Qs -O -K  >> $out

# -A is annotation interval in data units
# -L sets the limits
# -GDk5 every five km
awk '{print $1, $2, $3, 3}' thermo_res_init.csv | gmt psxy -i0,1,2,3s0.05 -S+0.15 -R -J -O -K  -W.8  >> $out
awk '{print $1, $2, $3, 3}' seis_res_init.csv | gmt psxy -i0,1,2,3s0.05 -Sd0.15 -R -J -O -K  -W.8 >> $out
gmt pscoast -W1/0.05 -Swhite -Df -J -R -K -O  -L169.25/-42.8/-42./50+l+u >> $out

gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt -Bx10f5 -By+l"Residual" -O -K  --FONT_ANNOT_PRIMARY=10p >> $out
#
awk '{print $1, $2}' mod_uplifts_init.txt | gmt psxy -i0,1 -Sc0.15 -R -J -O -K  -W.5 -Gwhite >> $out









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



rm -f mean.xyz track *.nc *.d gmt.conf

gmt psxy -R -J -T -O >> $out
gmt psconvert -Tf -A $out
#ps2raster -Tf -A map.ps
evince ${out%.*}.pdf
