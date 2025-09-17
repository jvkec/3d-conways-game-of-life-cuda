#ifndef TYPES_H
#define TYPES_H

struct GameOfLifeParams
{
    int birth_min = 14;
    int birth_max = 19;
    int survival_min = 14;
    int survival_max = 19;
    int width = 96;
    int height = 96;
    int depth = 96;
};

// TODO: add performance metrics n maybe grid configs in the future

#endif