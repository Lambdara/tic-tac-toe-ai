#lang racket

(require "racket-tic-tac-toe/tic-tac-toe.rkt")
(require "racket-neural-network/neural-network.rkt")
(provide ai-move train random-move random-legal-move)

(define (disp x)
  (displayln x)
  x)

(define (random-move)
  (random 9))

(define (random-legal-move game)
  (define board (vector-copy (send game get-board)))
  (for-each (lambda (index)
              (when (equal? 'nil
                            (vector-ref board index))
                (vector-set! board index index)))
            (range 9))
  (define length (vector-length (vector-filter number?
                                               board)))
  (define decision
    (vector-ref
     (vector-filter number?
                    board)
     (random length)))
  decision)

;; Used to extract which output scores highest and which score it has
(define (argmax-and-max items)
  (define (go items index maximum maximum-index)
    (if (null? items)
        (cons maximum-index maximum)
        (go (cdr items)
            (+ 1 index)
            (max (car items) maximum)
            (if (> (car items) maximum)
                index
                maximum-index))))
  (go (cdr items)
      1
      (car items)
      0))

(define (training-game nn chance gain)
  (define game (make-object tic-tac-toe%))
  (define cross-inputs (list))
  (define cross-outputs (list))
  (define cross-decisions (list))
  (define circle-inputs (list))
  (define circle-outputs (list))
  (define circle-decisions (list))
  (define winner 'nil)
  (define (go)
    (define board (send game get-board))
    ;; If this method is used after move was made to judge the result you want
    ;; to switch the players
    (define current-player (send game get-current-player))
    (define other-player (send game get-other-player))
    (define input
      (vector-append
       (vector-map
        (lambda (x)
          (if (equal? x current-player)
              1
              0))
        board)
       (vector-map (lambda (x)
                     (if (equal? x 'nil)
                         1
                         0))
                   board)
       (vector-map (lambda (x)
                     (if (equal? x other-player)
                         1
                         0))
                   board)))

    (send nn feedforward input)
    (define output (vector-copy (send nn get-output)))
    (define decision
      (car
       (argmax-and-max
        (map (lambda (x)
               (+ (* (random) chance)
                  (* (random)
                     x
                     (- 1 chance))))
             (vector->list output)))))
    (if (equal? 'cross (send game get-current-player))
        (begin (set! cross-inputs (cons (vector-copy input) cross-inputs))
               (set! cross-outputs (cons (vector-copy output) cross-outputs))
               (set! cross-decisions (cons decision cross-decisions)))
        (begin (set! circle-inputs (cons (vector-copy input) circle-inputs))
               (set! circle-outputs (cons (vector-copy output) circle-outputs))
               (set! circle-decisions (cons decision circle-decisions))))
    (send game move decision)
    (set! winner (send game get-winner))
    (when (equal? winner 'nil)
      (go)))
  (go)
  (define cross-score
    (cond ((equal? winner 'cross)
           1)
          ((equal? winner 'circle)
           0)
          (else 0.5)))
  (define circle-score
    (cond ((equal? winner 'circle)
           1)
          ((equal? winner 'cross)
           0)
          (else 0.5)))
  (define cross-moves (length cross-decisions))
  (define circle-moves (length circle-decisions))
  (define (learn inputs outputs decisions result gain)
    (when (not (null? inputs))
      (define input (car inputs))
      (define output (car outputs))
      (define decision (car decisions))
      (send nn feedforward input)
      (vector-set! output decision result)
      (send nn backpropagate gain output)
      (learn (cdr inputs) (cdr outputs) (cdr decisions) result (* gain 0.8))))
  (learn cross-inputs cross-outputs cross-decisions cross-score gain)
  (learn circle-inputs circle-outputs circle-decisions circle-score gain))

;; Defines move based on game and neural network
;; In deterministic mode, it will take the move with the highest score,
;; otherwise, it will take the scores to create a probability that the move
;; is chosen (twice the score -> twice the probability that it will be chosen).
(define (ai-move game nn (deterministic #f))
  (define current-player (send game get-current-player))
  (define other-player (send game get-other-player))

  ;; Creates input vector based on current board of length 27.
  ;; one for every square indicating whether current player has square;
  ;; one for every square indicating whether square is free;
  ;; one for every square indicating whether other player has square.
  (define (make-input)
    (define board (send game get-board))
    ;; If this method is used after move was made to judge the result you want
    ;; to switch the players
    (define current-player (send game get-current-player))
    (define other-player (send game get-other-player))
    (vector-append
     (vector-map
      (lambda (x)
        (if (equal? x current-player)
            1
            0))
      board)
     (vector-map (lambda (x)
                   (if (equal? x 'nil)
                       1
                       0))
                 board)
     (vector-map (lambda (x)
                   (if (equal? x other-player)
                       1
                       0))
                 board)))

  ;; Make input and feed to network, get copy of output
  (define input (make-input))
  (send nn feedforward input)
  (define output (vector-copy (send nn get-output)))

  ;; Either do something random or take move corresponding to highest output,
  ;; depending on random roll and chance parameter.
  (car
   (argmax-and-max
    (if deterministic
        (vector->list output)
        (map (lambda (x)
               (* (random)
                  x))
             (vector->list output))))))

(define (train nn total (gain 0.3))
  (define (go n)
    (when (> n 0)
      (training-game nn 0.2 gain)
      (go (sub1 n))))
  (go total))
