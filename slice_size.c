/**
 *
 * Copyright (c) 2020 Yuusuke Miyazaki
 *
 * This software is released under the MIT License.
 * http://opensource.org/licenses/mit-license.php
 *
 **/
#include <stdio.h>
#include <stdint.h>

#include "slice_table.h"

struct slice_table slice_tables[] ={
{0,0,173},
{8,0,165},
{16,0,224},
{24,0,215},
{32,0,220},
{40,0,186},
{48,0,205},
{56,0,183},
{64,0,223},
{72,0,205},
{80,0,196},
{88,0,111},
{96,0,289},
{104,0,171},
{112,0,216},
{0,1,148},
{8,1,155},
{16,1,211},
{24,1,179},
{32,1,182},
{40,1,238},
{48,1,260},
{56,1,221},
{64,1,170},
{72,1,216},
{80,1,194},
{88,1,174},
{96,1,232},
{104,1,154},
{112,1,230},
{0,2,160},
{8,2,140},
{16,2,152},
{24,2,151},
{32,2,128},
{40,2,186},
{48,2,221},
{56,2,269},
{64,2,246},
{72,2,204},
{80,2,255},
{88,2,172},
{96,2,195},
{104,2,179},
{112,2,182},
{0,3,147},
{8,3,123},
{16,3,130},
{24,3,141},
{32,3,140},
{40,3,146},
{48,3,219},
{56,3,325},
{64,3,334},
{72,3,236},
{80,3,256},
{88,3,198},
{96,3,179},
{104,3,160},
{112,3,184},
{0,4,173},
{8,4,149},
{16,4,137},
{24,4,142},
{32,4,115},
{40,4,146},
{48,4,181},
{56,4,260},
{64,4,396},
{72,4,258},
{80,4,241},
{88,4,176},
{96,4,167},
{104,4,146},
{112,4,173},
{0,5,193},
{8,5,155},
{16,5,130},
{24,5,134},
{32,5,115},
{40,5,136},
{48,5,141},
{56,5,160},
{64,5,235},
{72,5,298},
{80,5,334},
{88,5,264},
{96,5,200},
{104,5,177},
{112,5,171},
{0,6,164},
{8,6,153},
{16,6,119},
{24,6,136},
{32,6,112},
{40,6,107},
{48,6,113},
{56,6,136},
{64,6,176},
{72,6,248},
{80,6,227},
{88,6,302},
{96,6,241},
{104,6,200},
{112,6,204},
{0,7,152},
{8,7,136},
{16,7,160},
{24,7,150},
{32,7,135},
{40,7,117},
{48,7,120},
{56,7,140},
{64,7,149},
{72,7,170},
{80,7,192},
{88,7,224},
{96,7,220},
{104,7,209},
{112,7,214},
{0,8,134},
{8,8,160},
{16,8,158},
{24,8,152},
{32,8,132},
{40,8,123},
{48,8,112},
{56,8,121},
{64,8,174},
{72,8,155},
{80,8,164},
{88,8,209},
{96,8,204},
{104,8,165},
{112,8,170},
{0,9,144},
{8,9,136},
{16,9,136},
{24,9,173},
{32,9,142},
{40,9,124},
{48,9,116},
{56,9,121},
{64,9,140},
{72,9,157},
{80,9,153},
{88,9,194},
{96,9,205},
{104,9,202},
{112,9,167},
{0,10,140},
{8,10,131},
{16,10,129},
{24,10,150},
{32,10,110},
{40,10,115},
{48,10,117},
{56,10,118},
{64,10,105},
{72,10,121},
{80,10,140},
{88,10,146},
{96,10,176},
{104,10,183},
{112,10,194},
{0,11,181},
{8,11,137},
{16,11,139},
{24,11,162},
{32,11,138},
{40,11,153},
{48,11,121},
{56,11,117},
{64,11,117},
{72,11,103},
{80,11,128},
{88,11,137},
{96,11,161},
{104,11,182},
{112,11,209},
{0,12,161},
{8,12,127},
{16,12,128},
{24,12,141},
{32,12,145},
{40,12,117},
{48,12,111},
{56,12,93},
{64,12,100},
{72,12,109},
{80,12,93},
{88,12,126},
{96,12,147},
{104,12,216},
{112,12,215},
{0,13,154},
{8,13,119},
{16,13,128},
{24,13,150},
{32,13,149},
{40,13,126},
{48,13,100},
{56,13,112},
{64,13,117},
{72,13,119},
{80,13,139},
{88,13,160},
{96,13,184},
{104,13,182},
{112,13,219},
{0,14,148},
{8,14,122},
{16,14,126},
{24,14,123},
{32,14,167},
{40,14,123},
{48,14,96},
{56,14,97},
{64,14,95},
{72,14,98},
{80,14,117},
{88,14,121},
{96,14,152},
{104,14,206},
{112,14,198},
{0,15,198},
{8,15,197},
{16,15,130},
{24,15,142},
{32,15,321},
{40,15,136},
{48,15,125},
{56,15,117},
{64,15,127},
{72,15,116},
{80,15,120},
{88,15,125},
{96,15,158},
{104,15,191},
{112,15,174},
{0,16,194},
{8,16,196},
{16,16,111},
{24,16,193},
{32,16,302},
{40,16,122},
{48,16,106},
{56,16,121},
{64,16,116},
{72,16,95},
{80,16,139},
{88,16,113},
{96,16,169},
{104,16,138},
{112,16,151},
{0,17,194},
{8,17,200},
{16,17,189},
{24,17,196},
{32,17,210},
{40,17,137},
{48,17,108},
{56,17,116},
{64,17,127},
{72,17,102},
{80,17,153},
{88,17,143},
{96,17,136},
{104,17,138},
{112,17,141},
{0,18,192},
{8,18,198},
{16,18,200},
{24,18,195},
{32,18,192},
{40,18,151},
{48,18,132},
{56,18,96},
{64,18,124},
{72,18,136},
{80,18,137},
{88,18,138},
{96,18,128},
{104,18,116},
{112,18,122},
{0,19,195},
{8,19,200},
{16,19,199},
{24,19,196},
{32,19,196},
{40,19,207},
{48,19,122},
{56,19,110},
{64,19,116},
{72,19,118},
{80,19,115},
{88,19,139},
{96,19,147},
{104,19,146},
{112,19,127},
{0,20,197},
{8,20,197},
{16,20,194},
{24,20,196},
{32,20,196},
{40,20,196},
{48,20,183},
{56,20,109},
{64,20,114},
{72,20,114},
{80,20,106},
{88,20,115},
{96,20,127},
{104,20,134},
{112,20,153},
{0,21,196},
{8,21,200},
{16,21,198},
{24,21,200},
{32,21,196},
{40,21,195},
{48,21,198},
{56,21,125},
{64,21,124},
{72,21,126},
{80,21,107},
{88,21,109},
{96,21,145},
{104,21,160},
{112,21,146},
{0,22,198},
{8,22,199},
{16,22,196},
{24,22,192},
{32,22,198},
{40,22,196},
{48,22,194},
{56,22,198},
{64,22,131},
{72,22,104},
{80,22,116},
{88,22,116},
{96,22,132},
{104,22,129},
{112,22,128},
{0,23,195},
{8,23,197},
{16,23,197},
{24,23,199},
{32,23,199},
{40,23,200},
{48,23,198},
{56,23,199},
{64,23,164},
{72,23,123},
{80,23,267},
{88,23,127},
{96,23,122},
{104,23,147},
{112,23,137},
{0,24,197},
{8,24,199},
{16,24,194},
{24,24,198},
{32,24,199},
{40,24,199},
{48,24,198},
{56,24,198},
{64,24,206},
{72,24,83},
{80,24,319},
{88,24,189},
{96,24,189},
{104,24,144},
{112,24,153},
{0,25,195},
{8,25,197},
{16,25,198},
{24,25,194},
{32,25,199},
{40,25,198},
{48,25,200},
{56,25,200},
{64,25,200},
{72,25,205},
{80,25,199},
{88,25,206},
{96,25,202},
{104,25,137},
{112,25,153},
{0,26,195},
{8,26,197},
{16,26,195},
{24,26,195},
{32,26,198},
{40,26,198},
{48,26,200},
{56,26,200},
{64,26,194},
{72,26,197},
{80,26,200},
{88,26,196},
{96,26,193},
{104,26,198},
{112,26,139},
{0,27,194},
{8,27,199},
{16,27,198},
{24,27,196},
{32,27,195},
{40,27,198},
{48,27,198},
{56,27,200},
{64,27,200},
{72,27,196},
{80,27,195},
{88,27,196},
{96,27,199},
{104,27,147},
{112,27,140},
{0,28,199},
{8,28,196},
{16,28,193},
{24,28,198},
{32,28,200},
{40,28,199},
{48,28,200},
{56,28,198},
{64,28,199},
{72,28,198},
{80,28,199},
{88,28,192},
{96,28,198},
{104,28,223},
{112,28,150},
{0,29,195},
{8,29,200},
{16,29,195},
{24,29,195},
{32,29,199},
{40,29,198},
{48,29,198},
{56,29,196},
{64,29,198},
{72,29,198},
{80,29,199},
{88,29,199},
{96,29,200},
{104,29,198},
{112,29,210},
{0,30,198},
{8,30,198},
{16,30,192},
{24,30,205},
{32,30,191},
{40,30,198},
{48,30,199},
{56,30,199},
{64,30,199},
{72,30,199},
{80,30,197},
{88,30,199},
{96,30,198},
{104,30,199},
{112,30,197},
{0,31,198},
{8,31,195},
{16,31,182},
{24,31,223},
{32,31,196},
{40,31,193},
{48,31,200},
{56,31,198},
{64,31,196},
{72,31,199},
{80,31,197},
{88,31,198},
{96,31,197},
{104,31,198},
{112,31,199},
{0,32,196},
{8,32,197},
{16,32,198},
{24,32,198},
{32,32,191},
{40,32,197},
{48,32,197},
{56,32,199},
{64,32,200},
{72,32,197},
{80,32,200},
{88,32,196},
{96,32,199},
{104,32,200},
{112,32,200},
{0,33,194},
{8,33,196},
{16,33,199},
{24,33,197},
{32,33,197},
{40,33,212},
{48,33,200},
{56,33,194},
{64,33,200},
{72,33,199},
{80,33,200},
{88,33,199},
{96,33,196},
{104,33,196},
{112,33,196},
{0,34,199},
{8,34,194},
{16,34,200},
{24,34,197},
{32,34,192},
{40,34,197},
{48,34,199},
{56,34,200},
{64,34,197},
{72,34,196},
{80,34,199},
{88,34,199},
{96,34,195},
{104,34,199},
{112,34,193},
{0,35,200},
{8,35,200},
{16,35,199},
{24,35,200},
{32,35,196},
{40,35,192},
{48,35,196},
{56,35,197},
{64,35,198},
{72,35,200},
{80,35,199},
{88,35,200},
{96,35,200},
{104,35,196},
{112,35,197},
{0,36,197},
{8,36,199},
{16,36,196},
{24,36,196},
{32,36,197},
{40,36,197},
{48,36,214},
{56,36,198},
{64,36,198},
{72,36,194},
{80,36,196},
{88,36,199},
{96,36,198},
{104,36,196},
{112,36,200},
{0,37,178},
{8,37,195},
{16,37,193},
{24,37,198},
{32,37,200},
{40,37,199},
{48,37,227},
{56,37,198},
{64,37,193},
{72,37,197},
{80,37,193},
{88,37,198},
{96,37,196},
{104,37,198},
{112,37,199},
{0,38,189},
{8,38,195},
{16,38,190},
{24,38,200},
{32,38,199},
{40,38,199},
{48,38,216},
{56,38,198},
{64,38,199},
{72,38,198},
{80,38,198},
{88,38,200},
{96,38,193},
{104,38,198},
{112,38,199},
{0,39,196},
{8,39,197},
{16,39,195},
{24,39,196},
{32,39,196},
{40,39,197},
{48,39,134},
{56,39,289},
{64,39,198},
{72,39,200},
{80,39,197},
{88,39,196},
{96,39,199},
{104,39,198},
{112,39,200},
{0,40,192},
{8,40,200},
{16,40,208},
{24,40,198},
{32,40,198},
{40,40,198},
{48,40,113},
{56,40,290},
{64,40,196},
{72,40,197},
{80,40,199},
{88,40,195},
{96,40,199},
{104,40,199},
{112,40,194},
{0,41,189},
{8,41,197},
{16,41,195},
{24,41,188},
{32,41,199},
{40,41,197},
{48,41,160},
{56,41,266},
{64,41,198},
{72,41,200},
{80,41,193},
{88,41,191},
{96,41,200},
{104,41,194},
{112,41,199},
{0,42,200},
{8,42,198},
{16,42,199},
{24,42,189},
{32,42,200},
{40,42,198},
{48,42,210},
{56,42,195},
{64,42,200},
{72,42,199},
{80,42,200},
{88,42,203},
{96,42,196},
{104,42,195},
{112,42,200},
{0,43,193},
{8,43,199},
{16,43,192},
{24,43,195},
{32,43,198},
{40,43,198},
{48,43,187},
{56,43,196},
{64,43,197},
{72,43,200},
{80,43,225},
{88,43,183},
{96,43,224},
{104,43,200},
{112,43,194},
{0,44,194},
{8,44,195},
{16,44,203},
{24,44,199},
{32,44,195},
{40,44,196},
{48,44,191},
{56,44,198},
{64,44,210},
{72,44,191},
{80,44,197},
{88,44,190},
{96,44,236},
{104,44,203},
{112,44,202},
{0,45,133},
{8,45,256},
{16,45,199},
{24,45,197},
{32,45,203},
{40,45,199},
{48,45,197},
{56,45,197},
{64,45,204},
{72,45,199},
{80,45,192},
{88,45,173},
{96,45,243},
{104,45,185},
{112,45,208},
{0,46,200},
{8,46,198},
{16,46,193},
{24,46,201},
{32,46,206},
{40,46,195},
{48,46,200},
{56,46,199},
{64,46,189},
{72,46,195},
{80,46,191},
{88,46,195},
{96,46,191},
{104,46,206},
{112,46,234},
{0,47,112},
{8,47,282},
{16,47,200},
{24,47,197},
{32,47,129},
{40,47,280},
{48,47,195},
{56,47,196},
{64,47,195},
{72,47,194},
{80,47,193},
{88,47,196},
{96,47,200},
{104,47,196},
{112,47,221},
{0,48,92},
{8,48,307},
{16,48,194},
{24,48,182},
{32,48,185},
{40,48,196},
{48,48,241},
{56,48,200},
{64,48,200},
{72,48,198},
{80,48,199},
{88,48,197},
{96,48,196},
{104,48,197},
{112,48,186},
{0,49,138},
{8,49,261},
{16,49,192},
{24,49,188},
{32,49,134},
{40,49,278},
{48,49,191},
{56,49,197},
{64,49,199},
{72,49,195},
{80,49,193},
{88,49,197},
{96,49,198},
{104,49,231},
{112,49,200},
{0,50,131},
{8,50,264},
{16,50,199},
{24,50,195},
{32,50,103},
{40,50,296},
{48,50,191},
{56,50,199},
{64,50,198},
{72,50,192},
{80,50,198},
{88,50,198},
{96,50,199},
{104,50,201},
{112,50,224},
{0,51,135},
{8,51,263},
{16,51,196},
{24,51,203},
{32,51,145},
{40,51,243},
{48,51,190},
{56,51,184},
{64,51,199},
{72,51,200},
{80,51,199},
{88,51,198},
{96,51,198},
{104,51,191},
{112,51,247},
{0,52,120},
{8,52,274},
{16,52,200},
{24,52,197},
{32,52,176},
{40,52,207},
{48,52,220},
{56,52,204},
{64,52,189},
{72,52,200},
{80,52,196},
{88,52,166},
{96,52,198},
{104,52,198},
{112,52,195},
{0,53,196},
{8,53,197},
{16,53,170},
{24,53,222},
{32,53,208},
{40,53,171},
{48,53,234},
{56,53,162},
{64,53,233},
{72,53,196},
{80,53,185},
{88,53,157},
{96,53,160},
{104,53,298},
{112,53,193},
{0,54,196},
{8,54,197},
{16,54,187},
{24,54,210},
{32,54,186},
{40,54,199},
{48,54,225},
{56,54,149},
{64,54,245},
{72,54,194},
{80,54,197},
{88,54,142},
{96,54,168},
{104,54,298},
{112,54,161},
{0,55,138},
{8,55,256},
{16,55,196},
{24,55,200},
{32,55,130},
{40,55,258},
{48,55,221},
{56,55,152},
{64,55,233},
{72,55,198},
{80,55,197},
{88,55,152},
{96,55,224},
{104,55,234},
{112,55,160},
{0,56,172},
{8,56,162},
{16,56,256},
{24,56,208},
{32,56,170},
{40,56,119},
{48,56,308},
{56,56,188},
{64,56,217},
{72,56,196},
{80,56,197},
{88,56,162},
{96,56,239},
{104,56,201},
{112,56,182},
{0,57,181},
{8,57,124},
{16,57,286},
{24,57,190},
{32,57,195},
{40,57,172},
{48,57,221},
{56,57,217},
{64,57,211},
{72,57,192},
{80,57,206},
{88,57,143},
{96,57,257},
{104,57,197},
{112,57,191},
{0,58,179},
{8,58,98},
{16,58,313},
{24,58,186},
{32,58,177},
{40,58,176},
{48,58,235},
{56,58,206},
{64,58,222},
{72,58,190},
{80,58,202},
{88,58,200},
{96,58,183},
{104,58,208},
{112,58,191},
{0,59,186},
{8,59,137},
{16,59,260},
{24,59,208},
{32,59,184},
{40,59,168},
{48,59,185},
{56,59,267},
{64,59,191},
{72,59,197},
{80,59,216},
{88,59,177},
{96,59,181},
{104,59,181},
{112,59,231},
{0,60,199},
{8,60,187},
{16,60,169},
{24,60,236},
{32,60,206},
{40,60,157},
{48,60,221},
{56,60,177},
{64,60,243},
{72,60,193},
{80,60,174},
{88,60,221},
{96,60,154},
{104,60,139},
{112,60,316},
{0,61,200},
{8,61,172},
{16,61,220},
{24,61,204},
{32,61,200},
{40,61,191},
{48,61,132},
{56,61,275},
{64,61,196},
{72,61,199},
{80,61,153},
{88,61,234},
{96,61,149},
{104,61,189},
{112,61,284},
{0,62,196},
{8,62,178},
{16,62,193},
{24,62,192},
{32,62,234},
{40,62,195},
{48,62,167},
{56,62,245},
{64,62,197},
{72,62,195},
{80,62,187},
{88,62,202},
{96,62,138},
{104,62,218},
{112,62,259},
{0,63,199},
{8,63,198},
{16,63,188},
{24,63,193},
{32,63,201},
{40,63,196},
{48,63,194},
{56,63,196},
{64,63,193},
{72,63,198},
{80,63,226},
{88,63,207},
{96,63,144},
{104,63,218},
{112,63,244},
{0,64,199},
{8,64,195},
{16,64,164},
{24,64,226},
{32,64,190},
{40,64,220},
{48,64,194},
{56,64,199},
{64,64,198},
{72,64,197},
{80,64,198},
{88,64,173},
{96,64,166},
{104,64,203},
{112,64,258},
{0,65,200},
{8,65,198},
{16,65,172},
{24,65,200},
{32,65,199},
{40,65,213},
{48,65,213},
{56,65,197},
{64,65,200},
{72,65,195},
{80,65,199},
{88,65,192},
{96,65,183},
{104,65,205},
{112,65,234},
{0,66,200},
{8,66,199},
{16,66,190},
{24,66,197},
{32,66,193},
{40,66,197},
{48,66,192},
{56,66,193},
{64,66,198},
{72,66,198},
{80,66,194},
{88,66,235},
{96,66,167},
{104,66,145},
{112,66,296},
{0,67,190},
{8,67,191},
{16,67,190},
{24,67,197},
{32,67,212},
{40,67,200},
{48,67,195},
{56,67,200},
{64,67,198},
{72,67,196},
{80,67,207},
{88,67,220},
{96,67,189},
{104,67,204},
{112,67,192},
{0xff,0xff,0xffff}
};
