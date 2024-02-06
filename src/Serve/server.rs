use actix_web::{web, App, HttpResponse, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;

// Assuming GameState, Move, and Agent are defined elsewhere in your codebase.
#[derive(Serialize, Deserialize)]
struct MoveRequest {
    board_size: usize,
    moves: Vec<String>, // Assuming moves are serialized as strings.
}

// Dummy Agent trait for illustration.
trait Agent {
    fn select_move(&self, game_state: &GameState) -> Move;
    // Include any additional methods required by your agents.
}

// Dummy GameState and Move structs for illustration.
struct GameState; // Define this based on your game logic.
struct Move; // Define this based on your game logic.

// A map to store your bots, wrapped in a Mutex for thread-safe mutation.
struct BotMap {
    bots: Mutex<HashMap<String, Box<dyn Agent + Send + Sync>>>,
}

// Handler for the "select-move" route.
async fn select_move(
    bot_name: web::Path<String>,
    move_request: web::Json<MoveRequest>,
    bot_map: web::Data<BotMap>,
) -> impl Responder {
    let bot_map_lock = bot_map.bots.lock().unwrap();
    let agent = match bot_map_lock.get(&bot_name.into_inner()) {
        Some(agent) => agent,
        None => return HttpResponse::NotFound().finish(),
    };

    // Here, you would deserialize the MoveRequest into your game state
    // and apply the moves to reach the current state.
    let game_state = GameState; // Placeholder for game state reconstruction.

    let bot_move = agent.select_move(&game_state);

    // Serialize the move and return it as JSON.
    HttpResponse::Ok().json(serde_json::to_value(bot_move).unwrap())
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let bot_map = web::Data::new(BotMap {
        bots: Mutex::new(HashMap::new()),
    });

    // Initialize your bots here and add them to the `bot_map`.

    HttpServer::new(move || {
        App::new()
            .app_data(bot_map.clone())
            .route("/select-move/{bot_name}", web::post().to(select_move))
            // Define other routes and static file handling as needed.
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}