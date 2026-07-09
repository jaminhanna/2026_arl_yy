set -x

if test $# -lt 2
then
  echo usage: sh format_fred.sh path/to/FRED nthreads 1>&2
  exit 1
fi

if test ! -d $1
then
  echo bad path: $1 1>&2
  exit 1
fi

cd $1

if test $2 -lt 1
then
  echo bad number of threads: $2 1>&2
  exit 1
fi

nthreads=$2

# Create a label file for each
# event frame.

np=0

for i in train test
do
  for ((j=0; j<=230; j++))
  do
    if test -d $i/$j
    then
      for k in canonical/$i/images/event/Video_${j}_*
      do

        # Get the label filename and the
        # timestamp that is part of it.

        f=`echo $k | sed 's/.*\///
                          s/\.png/.txt/'`
        t=`echo $f | sed 's/.*_//
                          s/\.txt//'`

        if test ! -f canonical/$i/labels/event/$f
        then
          ( sed 's/\(\....\):/\1000:/
                 s/\(\.....\):/\100:/
                 s/\(\......\):/\10:/
                 s/\.//
                 s/://
                 s/,//g' $i/$j/coordinates.txt | grep "^$t " |

          # Normalize the bounding box coordinates
          # and grab the class "index" from the
          # class name.

          awk '{

            x1 = $2 / 1280
            y1 = $3 / 720
            x2 = $4 / 1280
            y2 = $5 / 720

            if (NF == 8) {
              label = $7 " " $8
            } else {
              label = $7 " " $8 " " $9
            }

            switch (label) {
            case "DarwinFPV cineape20":
              c = 0
              break
            case "DarwinFPV cineape20ger":
              c = 1
              break
            case "DJI Mini 2":
              c = 2
              break
            case "DJI Mini 3":
              c = 3
              break
            case "DJI Tello EDU":
              c = 4
              break
            }

            printf("%d %lf %lf %lf %lf\n", c, x1, y1, x2, y2)

          }' > canonical/$i/labels/event/$f ) &

          np=$((np+1))

          if test $np -eq $nthreads
          then
            wait
            np=0
          fi

        fi
      done
    fi
  done
done
wait

# Create a symbolic link for each RGB
# label file to the corresponding event
# label file.

np=0

for i in train test
do
  for ((j=0; j<=230; j++))
  do
    if test -d $i/$j
    then

      # The code gets the appropriate RGB label
      # filename for each event label filename.

      ( echo `ls canonical/$i/labels/event/Video_${j}_* |
              sed 's/_/_ /3
                   s/\.txt/ .txt/' | sort -n -k 2 | sed 's/ //g'` \
             `ls canonical/$i/images/rgb/Video_${j}_* |
              sed 's/.*\///
                   s/\.jpg/.txt/'` |

      # Construct and execute the correct
      # "ln -s" command.

      awk \
        -v d=`pwd` \
        -v i=$i \
      'BEGIN {
        system("set -x")
        prefix = "canonical/" $i "/labels/rgb"
      }

      {
        for (k = 1; k <= NF / 2; k++) {
          f = d "/" $k
          f2 = prefix "/" $(k + NF / 2)
          cmd = "ln -s " f " " f2
          system(cmd)
        }
      }

      END {
        system("set +x")
      }' ) &

      np=$((np+1))

      if test $np -eq $nthreads
      then
        wait
        np=0
      fi

    fi
  done
done
wait

set +x
