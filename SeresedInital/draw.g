set title "loss burnt time each para update without init"
set terminal png
set output "output_loss_for_test.png"
plot "long1000_short1_10.dat" title "burnT=1", "long1000_short10_10.dat" title "burnT=10", "long1000_short100_10.dat" title "burnT=100", "long1000_short1000_10.dat" title "burnT=1000"
set terminal aqua
