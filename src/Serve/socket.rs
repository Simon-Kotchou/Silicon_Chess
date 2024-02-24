use actix::{Actor, StreamHandler};
use actix_web_actors::ws;
use std::time::{Duration, Instant};
use actix_web::{web, Error, HttpRequest, HttpResponse};

/// Define a message that your WebSocket might need to handle.
/// This could be a chat message, game move, etc.
#[derive(Debug)]
pub struct Message {
    pub content: String,
}

/// Define your WebSocket actor
pub struct GameSocket {
    /// Client might have a unique identifier, for example, a user id
    pub id: usize,
    /// Last heartbeat from the client
    pub hb: Instant,
}

impl GameSocket {
    pub fn new(id: usize) -> Self {
        Self { 
            id,
            hb: Instant::now(),
        }
    }

    /// Function to handle the heartbeat mechanism
    fn hb(&self, ctx: &mut <Self as Actor>::Context) {
        ctx.run_interval(Duration::from_secs(5), |act, ctx| {
            if Instant::now().duration_since(act.hb) > Duration::from_secs(10) {
                // Heartbeat timed out
                println!("WebSocket Client {} heartbeat failed, disconnecting!", act.id);

                // Stop the actor's context
                ctx.stop();

                // No need to try to send a ping anymore
                return;
            }

            ctx.ping(b"");
        });
    }
}

/// Implement the Actor trait for GameSocket. This is where you define the context type and
/// create the actor.
impl Actor for GameSocket {
    type Context = ws::WebsocketContext<Self>;

    /// Method called when the actor is started.
    fn started(&mut self, ctx: &mut Self::Context) {
        self.hb(ctx);
    }
}

/// Implement the StreamHandler trait to handle incoming WebSocket messages.
impl StreamHandler<Result<ws::Message, ws::ProtocolError>> for GameSocket {
    fn handle(&mut self, msg: Result<ws::Message, ws::ProtocolError>, ctx: &mut Self::Context) {
        // Handle different types of WebSocket messages here (text, binary, ping/pong, close, etc.)
        match msg {
            Ok(ws::Message::Ping(msg)) => {
                self.hb = Instant::now();
                ctx.pong(&msg);
            }
            Ok(ws::Message::Pong(_)) => {
                self.hb = Instant::now();
            }
            Ok(ws::Message::Text(text)) => {
                // Handle text messages
                let message = Message { content: text };
                println!("Received message from {}: {:?}", self.id, message);
            }
            Ok(ws::Message::Binary(_bin)) => {
                // Handle binary data if necessary
            }
            Ok(ws::Message::Close(_)) => {
                // Close the connection
                ctx.close(None);
                ctx.stop();
            }
            _ => (),
        }
    }
}

/// Entry point for the WebSocket connection
pub async fn game_socket(req: HttpRequest, stream: web::Payload, path: web::Path<(usize,)>) -> Result<HttpResponse, Error> {
    let user_id = path.into_inner().0; // Assuming you're using user ID as a path parameter
    ws::start(GameSocket::new(user_id), &req, stream)
}