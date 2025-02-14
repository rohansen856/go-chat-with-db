package api

import (
	"fmt"

	"github.com/gentcod/nlp-to-sql/chat"
	db "github.com/gentcod/nlp-to-sql/internal/database"
	"github.com/gentcod/nlp-to-sql/token"
	"github.com/gentcod/nlp-to-sql/util"
	"github.com/gin-gonic/gin"
)

// Server serves HTTP requests for our banking service
type Server struct {
	config         util.Config
	store          db.Store
	tokenGenerator token.Generator
	websocket      *chat.WebSocketServer
	router         *gin.Engine
}

// NewServer creates a new HTTP server amd setup routing
func NewServer(config util.Config, store db.Store, websocket *chat.WebSocketServer) (*Server, error) {
	tokenGenerator, err := token.NewPasetoGenerator(config.TokenSymmetricKey)
	if err != nil {
		return nil, fmt.Errorf("cannot initialize token generator: %v", err)
	}

	server := &Server{
		config:         config,
		store:          store,
		tokenGenerator: tokenGenerator,
		websocket:      websocket,
	}

	server.setupRouter()

	return server, nil
}

func (server *Server) setupRouter() {
	router := gin.Default()

	v1Routes := router.Group("/api/v1")

	v1Routes.POST("/user/signup", server.createUser)
	v1Routes.POST("/user/login", server.loginUser)

	authRoutes := v1Routes.Group("/").Use((authMiddleware(server.tokenGenerator)))

	authRoutes.PATCH("/user/update", server.updateUser)

	// chain websocket server
	authRoutes.GET("/chat", server.websocket.HandleConnection)

	server.router = router
}

// Start runs HTTP server on a specific address
func (server *Server) Start(address string) error {
	return server.router.Run(address)
}

// apiErrorResponse returns a custom error response.
func apiErrorResponse(err error) gin.H {
	return gin.H{
		"status":  "error",
		"message": err.Error(),
	}
}
