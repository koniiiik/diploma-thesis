set autoscale
set xlabel "c"
set ylabel "coefficient"
#set key top right
#set xtics 1000000
set term tikz monochrome
set output "dpa-lower-bound-half.tikz.tex"
set xrange [1:1.3333]
f(c) = 1/2 + ((1 - 1/c) * log(2 - 2/c) + (1/2) * ((2/c) - 1) * log((2/c) - 1)) / log(2)

plot f(x) notitle with lines
