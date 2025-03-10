#ifndef DEVICEVACRATESSOVLER_H
#define DEVICEVACRATESSOVLER_H

#include "../src/lattice/lattice_types.h"
#include "../src/type_define.h"

#define NN_TOTAL 126
#define RE_SCALE_SIZE 1.2
#define STREAM_SIZE 4

struct dev_Vacancy {
    // LatticeTypes type;
    _type_lattice_id id;
    double rates[8];
};

struct dev_nnLattice {
    LatticeTypes type;
    _type_lattice_id id;
};

struct dev_event {
    _type_lattice_id from_id;
    _type_lattice_id to_id;
    LatticeTypes to_type;
};

struct dev_atom{
    _type_lattice_id atom_id;
    LatticeTypes::lat_type atom_type;
    _type_lattice_id atom12nn[14];
    LatticeTypes::lat_type atom12nn_type[14];

    _type_lattice_id atom1nn[8];
    LatticeTypes::lat_type atom1nn_type[8];
    _type_lattice_id atom2nn[6];
    LatticeTypes::lat_type atom2nn_type[6];
    int random_num1;
    double ranmov;

    int out_sector;
    int exchange;
    _type_lattice_id exchange_id;
    LatticeTypes::lat_type exchange_type;

    int numRe;
    int numSIA;
};

#endif /* DEVICEVACRATESSOVLER_H */