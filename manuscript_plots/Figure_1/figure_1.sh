#
# GMT code to plot seismicity and thermochron data for gcubed manuscript
# KM Jan 2020
#

out=Gcubed_fig_1.eps

gmt set FORMAT_GEO_MAP D
gmt set FORMAT_GEO_MAP D
gmt set PS_MEDIA A0
gmt set FONT_ANNOT_PRIMARY Helvetica
gmt set FONT_ANNOT_PRIMARY 12
gmt set FONT_LABEL Helvetica
gmt set LABEL_FONT_SIZE 7

#
# set -o nounset
# set -o errexit


# Define map characteristics
# Define your area
north=-42.5
south=-44.5
east=171.8
west=168.65

DEMdir="/home/kmichall/Desktop/topo"

proj='-JM6i'


# echo Make basemap ...
# # make a basemap
echo Plotting coast...
gmt pscoast -W1/0.05 -Df $proj -R$west/$east/$south/$north -K -Y16 -B0.5wsEn -L169.25/-42.8/-42./50+l+u -P > $out

gmt makecpt -Cgray -Z -T0/5000/200 -I > topo.cpt
echo Using this clipped grid ....
gmt grdimage -R -J $DEMdir/clipped_topo.grd -Ctopo.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out


echo Plotting lakes...
gmt psxy -R -J ../GMT_files/nz.gmt -W0.05,black -Gwhite -O -K >> $out

gmt pscoast -W1/0.05 -Df -J -R -K -O -Swhite -P -L169.25/-42.8/-42./50+l+u >> $out

echo Plotting faults...
gmt psxy -R -J ../GMT_files/activefaults.xy -Wgray20 -W1p -O -K >> $out

# gmt psxy -R -J $topodir/activefaults.xy -Wgray11 -W.8p -O -K >> $out


# gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cages.cpt  -Bx20f10 -By+l" Ages (Ma)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out




gmt makecpt -Cviridis -T0/20/1  > seis.cpt
#gmt makecpt -Cno_green -T0/20/1  > seis.cpt

gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cseis.cpt  -Bx5f2.5 -By+l" Hypocentral depth (km)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out



################################
echo Plotting seismicity data...
awk '{if ($17<=4) print $3, $2, $4, 1+$17}' ../GMT_files/hypoDD.reloc3 | gmt psxy -i0,1,2,3s0.035 -Sc -R -J \
-O -K  -W.25 -Cseis.cpt >> $out
awk '{if ($17>4) print $3, $2, $4, 1+$17}' ../GMT_files/hypoDD.reloc3 | gmt psxy -i0,1,2,3s0.035 -Sc -R -J \
-O -K  -W.25 -Cseis.cpt >> $out

echo Plotting GPS stations...
#awk '{print $3, $2}' ../GMT_files/GPS_sta_GEONET.txt |
#    gmt psxy -R -J -Sd.2 -W0.4p -Gdarkorange2 -O -K  >> $out

echo Plotting seismic stations...
awk '{print $3, $2}' ../GMT_files/sta_SAMBA.txt |
    gmt psxy -R -J -Si.2 -W0.4p -Gdarkorange2 -O -K  >> $out
awk '{print $3, $2}' ../GMT_files/sta_SAMBA_new.txt |
    gmt psxy -R -J -Si.2 -W0.4p -Gdarkorange2 -O -K  >> $out
awk '{print $3, $2}' ../GMT_files/sta_GEONET.txt |
    gmt psxy -R -J -St.2 -W0.4p -Gdarkorange2 -O -K  >> $out
awk '{print $3, $2}' ../GMT_files/sta_WIZARD.txt |
    gmt psxy -R -J -Si.2 -W0.4p -Gdarkorange2 -O -K  >> $out
awk '{print $3, $2}' ../GMT_files/sta_DFDP10.txt |
    gmt psxy -R -J -Si.2 -W0.4p -Gdarkorange2 -O -K  >> $out
awk '{print $3, $2}' ../GMT_files/sta_DFDP13.txt |
    gmt psxy -R -J -Si.2 -W0.4p -Gdarkorange2 -O -K  >> $out
awk '{print $3, $2}' ../GMT_files/sta_ALFA_08.txt |
    gmt psxy -R -J -Si.2 -W0.4p -Gdarkorange2 -O -K  >> $out


echo Plotting velocity of Pacific Plate relative to Australia...
gmt pstext -R -J -O -K -F+f10p,Helvetica,gray10+jBL+a26 >> $out << END
170.75 -44.05 39.5 mm/yr
END
gmt psxy -N -SV0.15i+e -Gblack -W2p -O -K -R -J >> $out << END
171.0 -44. 244 1.5
END


echo Creating legend...
# # construct legend
gmt pslegend <<END -R -J -Dx4.65i/0.1i+w1.3i/1.6i/TC -C0.1i/0.1i -F+gwhite+pthin -P -O -K >> $out
G -.01i
S .04i c .1i white 0.2p 0.18i Eq. locations
G .07i
S .04i t .11i gray30 0.2p 0.18i Perm. sites
G .07i
S .04i i .11i gray30 0.2p 0.18i Temp. sites
G .07i
S .04i d .11i darkorange2 0.2p 0.18i GPS
G .07i
S .04i s .08i black 0.2p 0.18i Towns
G .065i
S .04i - .14i red thick 0.18i Active fault
END

echo Plot scale ...
# gmt psscale -Dx1/9.5+o0/0.6i+w1.5i/0.08i+h+e -R -J -CFrance2.cpt -Bx2000f1000 -By+l"Topography (m)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out

##############################################################################
#Toponyms
##############################################################################
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


echo Plotting Lake names...
gmt pstext -R -J -O -K  -F+f8p,Helvetica,navy+jBL+a0 -Gwhite >> $out << END
170.5 -43.9 Lake
170.46 -43.95 Tekapo
170.12 -44.05 Lake
170.08 -44.1 Pukaki
169.71 -44.3 Lake
169.71 -44.35 Ohau
END

#Mount Cook
gmt psxy -R -J -Sx.3 -W1p -Gwhite -O -K  >> $out << END
170.1410417 -43.5957472
END
gmt pstext -R -J -O -K -F+f9p,Helvetica,gray10+jB  >> $out << END
# 170.17 -43.609 Aoraki/Mt Cook
170.065 -43.65 Aoraki/
169.985 -43.69 Mount Cook
170.05 -43.73
END







# Second map


echo Plotting coast...
gmt pscoast -W1/0.05 -Df $proj -R$west/$east/$south/$north -K -O -Y-14 -B0.5wSEn -L169.25/-43.5/-42./50+l+u -P >> $out

echo Using this clipped grid ....
# gmt grdimage -R -J $DEMdir/clipped_topo.grd -CFrance2.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out
gmt grdimage -R -J $DEMdir/clipped_topo.grd -Ctopo.cpt -I$DEMdir/SAMBA_relief.grd  -O -K >> $out

echo Plotting lakes...
gmt psxy -R -J ../GMT_files/nz.gmt -W0.05,black -Gwhite -O -K >> $out

gmt pscoast -W1/0.05 -Df -J -R -K -O -Swhite -L169.25/-43.5/-42./50+l+u -P >> $out
# gmt psxy -R -J $topodir/activefaults.xy -Wgray11 -W.8p -O -K >> $out
#gmt psxy -R -J ../GMT_files/activefaults.xy -Wlightred -W.8p -O -K >> $out
gmt psxy -R -J ../GMT_files/activefaults.xy -Wgray20 -W1p -O -K >> $out

gmt makecpt -Chot -T0/20/5  > ages_.cpt

# awk '{print $1, $2, $4, 1}' AFT_ages_inside_grid.dat |
# gmt psxy -i0,1,2,3s0.05 -Si.25 -R -J -O -K  -W.2 -Cages_.cpt >> $out
# awk '{print $1, $2, $4, 1}' AHe_ages.txt |
# gmt psxy -i0,1,2,3s0.05 -St.25 -R -J -O -K  -W.2 -Cages_.cpt >> $out

# gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cages_.cpt  -Bx10f5 -By+l" Ages (Ma)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out

echo Plotting thermochronometric data...
gmt makecpt -Cviridis -T0/10/1  > ages.cpt
gmt psscale -Dx1/11+o0/0.6i+w1.5i/0.08i+h+e -R -J -Cages.cpt  -Bx2f1 -By+l" Age (Ma)" -O -K --FONT_ANNOT_PRIMARY=10p >> $out

awk '{print 1, $1+0.035, $2+0.035, $1, $2}' ZFT_ages_inside_grid.dat > ZFT_age_lines.txt
awk '{print "> -Z" $1, "\n", $2, $3, "\n", $4, $5}' ZFT_age_lines.txt |
       gmt psxy -R -J -O -K -m -Wblack -W0.7p -P >> $out
awk '{print $1+0.035, $2+0.035, $4, 0.5}' ZFT_ages_inside_grid.dat |
gmt psxy -i0,1,2,3s0.05 -Sc.15 -R -J -O -K  -W.5,black -Cages.cpt >> $out
awk '{print $1, $2, $4, 1}' ZFT_ages_inside_grid.dat |
gmt psxy -i0,1,2,3s0.05 -Sc.1 -R -J -O -K -Gblack >> $out

# create file that has the two locations
awk '{print 1, $1-0.035, $2-0.035, $1, $2}' AFT_ages_inside_grid.dat > AFT_age_lines.txt
# plot a line connecting the two locations
awk '{print "> -Z" $1, "\n", $2, $3, "\n", $4, $5}' AFT_age_lines.txt |
       gmt psxy -R -J -O -K -m -Wdodgerblue -W0.7p -P >> $out
# plot changed locations
awk '{print $1-0.035, $2-0.035, $4, 0.5}' AFT_ages_inside_grid.dat |
gmt psxy -i0,1,2,3s0.05 -Ss.15 -R -J -O -K  -W.5,dodgerblue -Cages.cpt >> $out
# plot initial locations
awk '{print $1, $2, $4, 1}' AFT_ages_inside_grid.dat |
gmt psxy -i0,1,2,3s0.05 -Sc.1 -R -J -O -K -Gblack >> $out

# create file that has the two locations
awk '{print 1, $1-0.035, $2+0.035, $1, $2}' AHe_ages.txt > AHe_age_lines.txt
# plot a line connecting the two locations
#awk '{print "> -Z" $1, "\n", $2, $3, "\n", $4, $5}' AHe_age_lines.txt |
#       gmt psxy -R -J -O -K -m -Wred -W0.7p -P >> $out
# plot changed locations
#awk '{print $1-0.035, $2+0.035, $4, 1}' AHe_ages.txt |
#gmt psxy -i0,1,2,3s0.05 -St.20 -R -J -O -K  -W.2,red -Cages.cpt >> $out
# plot initial locations
awk '{print $1, $2, $4, 0.5}' AHe_ages.txt |
#gmt psxy -i0,1,2,3s0.05 -Ss.1 -R -J -O -K  -Gblack >> $out
gmt psxy -i0,1,2,3s0.05 -St.15 -R -J -O -K  -W.5,dodgerblue -Cages.cpt >> $out
awk '{print $1, $2, $4, 0.5}' ZHe_ages_inside_grid.dat |
gmt psxy -i0,1,2,3s0.05 -Si.15 -R -J -O -K  -W.5,black -Cages.cpt >> $out


# # ZFT ages
# # awk '{print $1, $2, $4, 1}' ZFT_ages_inside_grid.dat |
# # gmt psxy -i0,1,2,3s0.05 -Ss.25 -R -J -O -K  -W.2 -Cages.cpt >> $out
# awk '{print $1, $2}' ZFT_ages_inside_grid.dat |
# gmt psxy -Sc0.05 -R -J -O -K -Gblack -W.2p >> $out
# # plot lines
# awk '{print "> -Z" $4, "\n", $1, $2, "\n", $1 - 0.05 , $2 - 0.05}' ZFT_ages_inside_grid.dat |
# gmt psxy -R -J -O -K -m  -Wthinnest,red -P >> $out
# # plot ages
# awk '{print $1-0.05, $2-0.05, $4}' ZFT_ages_inside_grid.dat |
# gmt pstext -R -J -O -K -F+f8p,Helvetica,gray10+jB -Glightred -W.1p,red >> $out
#
# # ZHe ages
# awk '{print $1, $2}' ZHe_ages_inside_grid.dat |
# gmt psxy -Sc0.05 -R -J -O -K -Gblack -W.2p >> $out
# # plot lines
# awk '{print "> -Z" $4, "\n", $1, $2, "\n", $1 + 0.05 , $2 + 0.05}' ZHe_ages_inside_grid.dat |
# gmt psxy -R -J -O -K -m  -Wthinnest,red -P >> $out
# # plot ages
# awk '{print $1+0.05, $2+0.05, $4}' ZHe_ages_inside_grid.dat |
# gmt pstext -R -J -O -K -F+f8p,Helvetica,gray10+jB -Gwhite -W.1p,red >> $out
#
# # AHe ages
# awk '{print $1, $2}' AHe_ages.txt |
# gmt psxy -Sc0.05 -R -J -O -K -Gblack -W.2p >> $out
# # plot lines
# awk '{print "> -Z" $4, "\n", $1, $2, "\n", $1 , $2 - 0.05}' AHe_ages.txt |
# gmt psxy -R -J -O -K -m  -Wthinnest,black -P >> $out
# # plot ages
# awk '{print $1, $2-0.05, $4}' AHe_ages.txt |
# gmt pstext -R -J -O -K -F+f8p,Helvetica,gray10+jB -Gwhite -W.1p,black >> $out
#
# # AFT ages
# awk '{print $1, $2}' AFT_ages_inside_grid.dat |
# gmt psxy -Sc0.05 -R -J -O -K -Gblack -W.2p >> $out
# # plot lines
# awk '{print "> -Z" $4, "\n", $1, $2, "\n", $1 + 0.05 , $2 - 0.05}' AFT_ages_inside_grid.dat |
# gmt psxy -R -J -O -K  -Wthinnest,black -P >> $out
# # plot ages
# awk '{print $1+0.05, $2-0.05, $4}' AFT_ages_inside_grid.dat |
# gmt pstext -R -J -O -K -F+f8p,Helvetica,gray10+jB -Glightgray -W.1p,black >> $out

echo Creating legend...
# construct legend
gmt pslegend <<END -R -J -Dx3.6i/0.06i+w2.35i/1.1i/TC -C0.1i/0.1i -F+gwhite+pthin -P -O -K >> $out
G -.01i
S .04i s .11i white 0.2p 0.18i Apatite Fission Track (AFT)
G .07i
S .04i t .11i white 0.2p 0.18i Apatite U-Th/He (AHe)
G .07i
S .04i c .11i white 0.2p 0.18i Zircon Fission Track (ZFT)
G .07i
S .04i i .11i white 0.2p 0.18i Zircon U-Th/He (ZHe)
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

#Mount Cook
gmt psxy -R -J -Sx.3 -W1p -Gwhite -O -K  >> $out << END
170.1410417 -43.5957472
END

#--------------------------------------------------------
# Inset map of New Zealand showing study area
# #--------------------------------------------------------
# echo Plotting inset ...
# echo ...
# region2="-R165/180/-48/-34."
# projection2="-JM4"
# boundaries2="-B80nsew"
#
# gmt psbasemap $region2 $projection2 $boundaries2 -X0.0255 -Y8.285 -O -K >> $out
#
# # grdimage -R -J $topodir/100m_dem_wgs84.grd -Ctopo.cpt -O -K >> $out
# # grdimage -R -J $topodir/SI_100m_dem_wgs84.grd -Ctopo.cpt -O -K >> $out
#
#
# # pscoast -R -J -Df -W1/0.05 -Swhite  -L176.1/-47/-47/400+l -O -K >> $out
# gmt pscoast -R -J -Df -W1/0.05p -Swhite -L176.1/-47/-47/400+l -O -K >> $out
# gmt psxy -R -J $topodir/nz.gmt -W0.01,black -G216/242/254 -O -K >> $out
#
# gmt psxy -R -J $topodir/PB_UTIG_Transform.xy -Sf0.5c/0.03i+l+t -Gblack -W -O -K  >> $out
# # psxy -R -J $topodir/Alpine_fault.xy -Sf0.5c/0.03i+l+t -Gred -W -O -K  >> $out
#
# gmt psxy -R -J $topodir/PB_UTIG_Transform.xy -Sf2c/0.1i+r+s+o1 -Gblack -W -O -K  >> $out
# # psxy $topodir/qmap_faults.gmt -R -J -V -O -K -W0.5,black -m >> $out
#
# echo Plot study area in the inset...
# # #study area
# gmt psxy -R -J -Wthinner,red -O -K  >> $out << END
# # 169.5 -44.9
# # 172   -43.5
# # 171   -42.5
# # 168.5 -43.8
# # 169.5 -44.9
# 168.65 -44.2
# 168.65 -42.5
# 171.8 -42.5
# 171.8  -44.2
# 168.65 -44.2
# END
#
#
# gmt pstext -R -J -O -K -F+f10p,Helvetica,gray10+jBL >> $out << END
# 176.1 -46.5 PAC
# 167   -38 AUS
# END
#
#
# gmt pstext -R -J -O -K  -F+f10p,Helvetica,gray10+jBL+a32  >> $out << END
# 169.0 -43.5 AF
# END
# gmt pstext -R -J -O -K  -F+f10p,Helvetica,gray10+jBL+a65  >> $out << END
# 178 -40 HT
# END
#
# gmt pstext -R -J -O -K  -F+f10p,Helvetica,gray10+jBL  >> $out << END
# 166 -47.5 PT
# END
# gmt pstext -R -J -O -K -F+f10p,Helvetica,gray10+jBL+a15 >> $out << END
# 175 -44.2 39.8 mm/yr
# END
# gmt psxy -Sv0.15i+ea -Wthin -GBlack -O -K -R -J >> $out << END
# 177 -44.2 -165 1.5
# END





rm -f mean.xyz track *.nc *.d gmt.conf

gmt psxy -R -J -T -O >> $out
gmt psconvert -Tf -A $out
#ps2raster -Tf -A map.ps
evince ${out%.*}.pdf
