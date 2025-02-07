/* Source the CPM module (cpm-cjs version because this is a node script).*/
let CPM = require("../../build/artistoo-cjs.js")

let config = {
    ndim : 2,
    field_size : [50,50],
    conf : {
        T : 20,			// CPM temperature				
        // Adhesion parameters:
        J: [[0,20], [20,100]] ,
        // VolumeConstraint parameters
        LAMBDA_V : [0,50],	// VolumeConstraint importance per cellkind
        V : [0,500]		// Target volume of each cellkind
    },
    simsettings : {
        NRCELLS : [1],
        RUNTIME : 500,
        CANVASCOLOR : "eaecef",
        zoom : 4
    }
}

// let sim
// function initialize(){
//     sim = new CPM.Simulation( config )
// }
//         // Simulation code here.
// function step(){
//     sim.step()
//     requestAnimationFrame( step )
// }

let sim = new CPM.Simulation( config )
sim.run()