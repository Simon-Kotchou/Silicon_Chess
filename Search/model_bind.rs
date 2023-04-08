use std::os::raw::c_char;
use std::ffi::{CStr, CString};
use tch::{CModule, Device, Kind, Tensor};

pub mod mcts; // Include your MCTS code here

#[no_mangle]
pub extern "C" fn new_mcts(initial_state: *const c_char, max_iterations: u64, exploration_constant: f64) -> *mut mcts::MCTS {
    let initial_state_cstr = unsafe { CStr::from_ptr(initial_state) };
    let initial_state_str = initial_state_cstr.to_str().unwrap();
    let initial_state = GameState::from_str(initial_state_str); // Implement the from_str function for your GameState struct

    let zobrist = mcts::Zobrist::new(seed);
    let mcts = mcts::MCTS::new(initial_state, max_iterations, exploration_constant, zobrist);
    Box::into_raw(Box::new(mcts))
}

#[no_mangle]
pub extern "C" fn search(mcts_ptr: *mut mcts::MCTS) -> *mut c_char {
    let mcts = unsafe { &mut *mcts_ptr };
    let result = mcts.search();

    match result {
        Some(game_state) => {
            let state_str = game_state.to_string(); // Implement the ToString trait for your GameState struct
            CString::new(state_str).unwrap().into_raw()
        },
        None => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn free_string(s: *mut c_char) {
    unsafe {
        if s.is_null() { return }
        CString::from_raw(s)
    };
}


pub struct TorchModel {
    model: CModule,
}

impl TorchModel {
    pub fn new(model_path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let model = tch::CModule::load(model_path)?;
        Ok(TorchModel { model })
    }

    pub fn infer(&self, input: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let output = self.model.forward_ts(&[input])?;
        Ok(output)
    }
}