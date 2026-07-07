#!/bin/sh
# get_fred.sh: Download the FRED dataset to the directory indicated by
# the first argument and using the number of threads indicated by the
# second argument.

set -x

if test $# -lt 2
then
  echo usage: sh get_fred.sh path nthreads 1>&2
  exit 1
fi

if test ! -d $1
then
  echo bad path: $1 1>&2
  exit 1
fi

if test $2 -le 0
then
  echo bad number of threads: $2 1>&2
  exit
fi

np=0
path=$1
nthreads=$2
prefix=https://huggingface.co/datasets
prefix=$prefix/GabrieleMagrini/FRED/resolve/main

cd $path

mkdir FRED FRED/train FRED/test

cd FRED/train

for i in 0 1 2 3 4 5 6 7 10 11 13 \
         14 15 16 17 18 19 22 23 24 25 26 \
         27 28 30 31 32 33 35 36 37 38 40 \
         42 43 44 45 47 48 49 50 52 53 54 \
         55 56 57 58 59 61 62 66 67 68 69 \
         70 71 72 73 74 75 76 77 78 79 81 \
         82 84 85 86 87 88 89 91 92 94 95 \
         96 97 98 99 100 101 102 103 104 105 106 \
         107 108 109 112 113 116 117 118 119 120 121 \
         122 123 124 125 126 130 131 132 133 134 135 \
         136 137 140 141 143 144 145 146 147 148 149 \
         150 151 155 156 157 158 159 160 161 164 165 \
         166 168 169 170 171 172 173 174 175 176 177 \
         180 181 182 183 184 185 186 188 189 190 191 \
         192 193 197 198 199 200 201 203 204 205 206 \
         207 208 209 211 212 213 214 215 216 218 220 \
         221 222 224 225 226 227 228 229
do
  if test ! -d $i -a ! -f $i.zip
  then
    ( wget $prefix/train/$i.zip ) &
    np=$((np+1))
    if test $np -eq $nthreads
    then
      wait
      np=0
    fi
  fi
done
wait

np=0

cd ../test

for i in 8 9 12 20 21 29 34 39 41 46 51 \
         60 63 64 65 80 83 90 93 110 111 114 \
         115 127 128 129 138 139 142 152 153 154 162 \
         163 167 178 179 187 194 195 196 202 210 217 \
         219 223 230
do
  if test ! -d $i -a ! -f $i.zip
  then
    ( wget $prefix/test/$i.zip ) &
    np=$((np+1))
    if test $np -eq $nthreads
    then
      wait
      np=0
    fi
  fi
done
wait

np=0

cd ..

# Unzip and fix filenames.

for i in train test
do
  cd $i
  for ((j=0; j<=230; j++))
  do
    if test ! -d $j -a -f $j.zip
    then
      ( if unzip $j.zip
      then
        if test $j = 68
        then
          cd $j/RGB
          for k in *
          do
            mv $k `echo $k | sed 's/^/Video_68_/'`
          done
          cd ../..
        fi
        cd $j/Event/Frames
        for k in `ls | grep -v frame`
        do
          mv $k `echo $k | sed 's/_/_frame_/2'`
        done
        cd ../../..
      else
        rm $j.zip
      fi ) &
      np=$((np+1))
      if test $np -eq $nthreads
      then
        wait
        np=0
      fi
    fi
  done
  cd ..
done
wait

set +x
