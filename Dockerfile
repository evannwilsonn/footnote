# Build the Go binary, then copy it into a small runtime image.
FROM golang:1.24-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /footnote ./cmd/footnote

FROM alpine:3.20
RUN adduser -D -u 10001 footnote
WORKDIR /app
COPY --from=build /footnote /usr/local/bin/footnote
COPY docs ./docs
USER footnote
EXPOSE 8080
ENTRYPOINT ["footnote"]
CMD ["serve", "-addr", ":8080"]
