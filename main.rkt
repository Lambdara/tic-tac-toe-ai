#lang racket

;; This file defines a few ease-of-use methods for training the ai and playing
;; against it, defining a single neural-network at the start;

(require racket/date)

(require "racket-tic-tac-toe/tic-tac-toe.rkt")
(require "racket-neural-network/neural-network.rkt")
(require "ai.rkt")

;; The default neural network
(define nn (make-object neural-network% 27 81 9))

;; For setting the neural network to a different topology
(define (set-nn-topology! . topology)
  (set! nn (apply (curry make-object neural-network%) topology)))

;; Shorthands for training the neural network
(define (train-nn n (gain 0.3))
  (train nn n 0.3))

(define (test-vs-random n)
  (define ai-score 0)
  (define (play amount)
    (define game (make-object tic-tac-toe%))
    (define winner 'nil)
    (define ai-token (list-ref '(cross circle) (random 2)))
    (define random-token (if (equal? ai-token 'cross)
                             'circle
                             'cross))
    (define (move)
      (if (equal? (send game get-current-player)
                  random-token)
          (send game move (random-move))
          (send game move (ai-move game nn #t))))
    (define (go)
      (if (equal? winner 'nil)
          (begin (move)
                 (set! winner (send game get-winner))
                 (go))
          (when (equal? winner ai-token)
            (set! ai-score (add1 ai-score)))))
    (go)
    (when (> amount 1)
      (play (sub1 amount))))
  (play n)
  (displayln (string-append "AI won: "
                            (number->string (exact->inexact (/ ai-score n)))))
  (exact->inexact (/ ai-score n)))


(define (test-vs-random-legal n)
  (define ai-score 0)
  (define (play amount)
    (define game (make-object tic-tac-toe%))
    (define winner 'nil)
    (define ai-token (list-ref '(cross circle) (random 2)))
    (define random-token (if (equal? ai-token 'cross)
                             'circle
                             'cross))
    (define (move)
      (if (equal? (send game get-current-player)
                  random-token)
          (send game move (random-legal-move game))
          (send game move (ai-move game nn #t))))
    (define (go)
      (if (equal? winner 'nil)
          (begin (move)
                 (set! winner (send game get-winner))
                 (go))
          (when (equal? winner ai-token)
            (set! ai-score (add1 ai-score)))))
    (go)
    (when (> amount 1)
      (play (sub1 amount))))
  (play n)
  (displayln (string-append "AI won: "
                            (number->string (exact->inexact (/ ai-score n)))))
  (exact->inexact (/ ai-score n)))

;; Play a game, human-player should be 'cross or 'circle (or 'nil for ai vs. ai)
(define (play human-player)
  (define game (make-object tic-tac-toe%))
  (define (display-board)
    (define (symbol x)
      (cond ((equal? x 'circle) "O")
            ((equal? x 'cross) "X")
            (else " ")))
    (define board (vector-map symbol (vector-copy (send game get-board))))
    (displayln (vector-take board 3))
    (displayln (vector-take (vector-take-right board 6) 3))
    (displayln (vector-take-right board 3)))
  (display-board)
  (define (move-by-ai)
    (send game move (ai-move game nn #t)))
  (define (move-by-human index)
    (send game move index))
  (define (go)
    (define current-player (send game get-current-player))
    (if (equal? current-player human-player)
        (begin
          (displayln "Enter move [0-8]:")
          (move-by-human (read)))
        (move-by-ai))
    (displayln "Current board:")
    (display-board)
    (if (equal? (send game get-winner)
                'nil)
        (go)
        (begin (display "Winner of the game: ")
               (displayln (send game get-winner))
               (send game get-winner))))
  (go))

(define gain-param 0.7)
(define (go path)
  (displayln "Versus random...")
  (display-to-file
   (test-vs-random 1000)
   path
   #:exists 'append)
  (display-to-file
   ","
   path
   #:exists 'append)
  (displayln "Versus random-legal...")
  (display-to-file
   (test-vs-random-legal 1000)
   path
   #:exists 'append)
  (display-to-file
   "\n"
   path
   #:exists 'append)
  (displayln (string-append
              "Training 1000 games at "
              (number->string gain-param)
              "..."))
  (train-nn 1000 gain-param)
  (set! gain-param (* gain-param 0.95))
  (go path))
