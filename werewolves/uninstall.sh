#!/bin/bash

# -- delete players --
# del user
for i in {0..15}; do 
    userdel -f player$i; 
    rm -rf /home/player$i
done
# del grp
for i in {0..15}; do 
    groupdel -f player$i; 
done
# del home
for i in {0..15}; do 
    rm -rf /home/player$i
done

# -- delete moderator --
userdel -f moderator
groupdel -f moderator
rm -rf /home/moderator
