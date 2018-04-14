#ifndef FUNCS_H
#define	FUNCS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "serial.h"

typedef enum {RGB, GREY} color_t;

void setParams(int, char **, char **, int *, int *, color_t *);

#endif