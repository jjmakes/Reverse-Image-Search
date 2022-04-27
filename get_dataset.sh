#!/bin/sh

LFW="http://vis-www.cs.umass.edu/lfw/lfw.tgz"
MD5SUMS=static/md5sums.txt

wget -O lfw.tgz $LFW && md5sum -c $MD5SUMS && tar -xzf lfw.tgz
